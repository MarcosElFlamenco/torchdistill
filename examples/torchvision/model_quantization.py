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


def val_top1_accuracy_wrapper(model,quantized_model,config,args):

    dataset_dict = config["datasets"] 
    val_data_loader_config = config['quantize']["val_data_loader"]
    val_data_loader = build_data_loader(dataset_dict[val_data_loader_config['dataset_id']],val_data_loader_config,False)
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: 
        device = torch.device("cpu")
    logger.info(f"Evaluating original model")
    val_top1_accuracy_o = evaluate(model, val_data_loader, device, None, False,log_freq=config["quantize"]["log_freq"],header="Validation")
    logger.info(f'Evaluating quantized model')
    val_top1_accuracy_q = evaluate(quantized_model, val_data_loader, device, None, False,log_freq=config["quantize"]["log_freq"],header="Validation")


def get_argparser():
    parser = argparse.ArgumentParser(description='Knowledge distillation for image classification models')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--type', required = True, help='quantization type')
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

def load_model(model_config,args):
    model = resnet.resnet(20,10,False,False)
    ckpt_path = model_config["src_ckpt"]
    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    ckpt = torch.load(ckpt_path,weights_only=False,map_location=device)
    state_dict = ckpt["model"]
    model.load_state_dict(state_dict)
    return model

def format_size(size_bytes):
    """
    Format size in bytes to human-readable format (KB, MB, GB)
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def quantize(model,args,config):

    calibration_data_loader_config = config["test"]["test_data_loader"]

    dataset_dict = config['datasets']
    calibration_data_loader = build_data_loader(dataset_dict[calibration_data_loader_config['dataset_id']],calibration_data_loader_config,False)

    start_time = time.time()
    logger.info(f"Quantizing model with option: {args.type} ...")
    if args.type == "static":
        logger.info("WARNING static quantization is not supported as of rn")
        if args.fuse:
            print("Fusing model")
            model_fused = torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']])
        else:
            print('Model will not be fused')
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
    elif args.type == "dynamic":
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear, torch.nn.LSTM}, 
            dtype=torch.qint8
        )
    else:
        print(f"quantization type {args.type} is not recognized")
    logger.info(f"Model is quantized")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Quantization time {}'.format(total_time_str))
    dst_ckpt = config["model"]["dst_ckpt"]
    torch.save(quantized_model, dst_ckpt)
    # After quantization is complete:
    logger.info(f"Original model size: {format_size(os.path.getsize(config["model"]["src_ckpt"]))}")
    logger.info(f"Quantized model size: {format_size(os.path.getsize(config["model"]["dst_ckpt"]))}")
    logger.info(f"Successfully saved model to {dst_ckpt}")
    return quantized_model
 
def main(args):
    set_basic_log_config()

    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))

    # Load your pre-trained model
    model_config = config["model"]
    model = load_model(model_config,args).eval()

    # Use 'qnnpack' for ARM CPUs: torch.quantization.get_default_qconfig('qnnpack')
    #QUANTIZING
    if not args.test_only:
        quantized_model = quantize(model,args,config)
    else:
        quantized_model = torch.load(model_config["dst_ckpt"],weights_only=False)
        logger.info(f"Retrieved quantized model from {model_config["dst_ckpt"]}")

    quantized_model.eval()

    val_top1_accuracy_wrapper(model,quantized_model,config,args)
    #val_top1_accuracy_wrapper(quantized_model,config,dataset_dict,args)

if __name__ == "__main__":
    argparser = get_argparser()
    main(argparser.parse_args())