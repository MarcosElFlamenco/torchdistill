import torch
from torchdistill.models.classification import resnet
from torchdistill.datasets.util import build_data_loader
from torchdistill.common import file_util, yaml_util, module_util
import os
from torch.nn import DataParallel
import argparse
from torch.nn.parallel import DistributedDataParallel
from torchdistill.misc.log import set_basic_log_config, setup_log_file, SmoothedValue, MetricLogger
from examples.torchvision.change_targets import transform_targets
from torchdistill.common.constant import def_logger
import time
import datetime

logger = def_logger.getChild(__name__)

def get_argparser():
    parser = argparse.ArgumentParser(description='Knowledge distillation for image classification models')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('-test_only', action='store_true', help='only test the models')
    parser.add_argument('-fuse', action='store_true', help='fuse the layers')
    parser.add_argument('-cuda', action='store_true', help='run on cuda')

    return parser

def compute_accuracy(outputs, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, preds = outputs.topk(maxk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets[None])
        result_list = []
        for k in topk:
            correct_k = corrects[:k].flatten().sum(dtype=torch.float32)
            result_list.append(correct_k * (100.0 / batch_size))
        return result_list


@torch.inference_mode()
def evaluate(model, data_loader, device, device_ids, distributed, log_freq=1000, title=None, header='Test:',target_classes=None):
    model.to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=device_ids)
    elif device.type.startswith('cuda'):
        model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        if target_classes != None:
            target = transform_targets(target)
        target = target.to(device, non_blocking=True)
        output = model(image)
        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        # FIXME need to take into account that the datasets
        # could have been padded in distributed setup
        batch_size = image.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    return metric_logger.acc1.global_avg

def load_model_eval(model_config):
    model = resnet.resnet(20,10,False,False)
    state_dict = model_config["src_ckpt"]
    model.load_state_dict(torch.load(state_dict,weights_only=False)["model"])
    model.eval()
    return model


def main(args):
    
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    # Load your pre-trained model

    model_config = config["model"]
    model = load_model_eval(model_config)
   # Use 'qnnpack' for ARM CPUs: torch.quantization.get_default_qconfig('qnnpack')

    calibration_data_loader_config = config["test"]["test_data_loader"]

    dataset_dict = config['datasets']
    calibration_data_loader = build_data_loader(dataset_dict[calibration_data_loader_config['dataset_id']],calibration_data_loader_config,False)

    
    #QUANTIZING    
    if not args.test_only:
        start_time = time.time()
        logger.info("Quantizing model...")
        if args.fuse:
            model_fused = torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']])
        else:
            model_fused = model
        # Set up quantization configuration
        model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # for x86 CPUs

        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model_fused)
 
        # Calibrate with sample data
        with torch.no_grad():
            for data, _ in calibration_data_loader:
                model_prepared(data)

        quantized_model = torch.quantization.convert(model_prepared)
        logger.info(f"Model is quantized")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Quantization time {}'.format(total_time_str))
 
    else:
        quantized_model = torch.load(model_config["dst_ckpt"],weights_only=False)
        print(f"retrieved quantized model from {model_config["dst_ckpt"]}")
    print(f"quantized model is {quantized_model.type}")
    quantized_model.eval()

    # After quantization is complete:
    #scripted_quantized_model = torch.jit.script(quantized_model)
    #scripted_quantized_model.save('quantized_model_scripted.pt')

    # Later, to reload:
    #loaded_model = torch.jit.load('quantized_model_scripted.pt')
    #loaded_model.eval()

    val_data_loader_config = config['quantize']["val_data_loader"]
    val_data_loader = build_data_loader(dataset_dict[val_data_loader_config['dataset_id']],val_data_loader_config,False)

    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: 
        device = torch.device("cpu")
    #val_top1_accuracy = evaluate(quantized_model, val_data_loader, device, None, False,log_freq=config["quantize"]["log_freq"],header="Validation")

    # Save quantized model
    if not args.test_only:
        dst_ckpt = config["model"]["dst_ckpt"]
        print(f'saving model')
        torch.save(quantized_model, dst_ckpt)
        print("model saved")

if __name__ == "__main__":
    argparser = get_argparser()
    main(argparser.parse_args())