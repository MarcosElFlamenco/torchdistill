import argparse
import datetime
import os
import time

import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from torchdistill.common import file_util, yaml_util, module_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, init_distributed_mode, load_ckpt, save_ckpt, set_seed, \
    import_dependencies
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import get_training_box
from torchdistill.datasets.util import build_data_loader
from torchdistill.misc.log import set_basic_log_config, setup_log_file, SmoothedValue, MetricLogger
from torchdistill.models.official import get_image_classification_model
from torchdistill.models.registry import get_model

import wandb
import examples.torchvision.bucket_interactions as bi

logger = def_logger.getChild(__name__)


def get_argparser():
    parser = argparse.ArgumentParser(description='Knowledge distillation for image classification models')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--run_log', help='log file path')
    parser.add_argument('--project_name', default = "default classification project",help='project name')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--seed', type=int, help='seed in random number generator')
    parser.add_argument('-disable_cudnn_benchmark', action='store_true', help='disable torch.backend.cudnn.benchmark')
    parser.add_argument('-test_only', action='store_true', help='only test the models')
    parser.add_argument('-student_only', action='store_true', help='test the student model only')
    parser.add_argument('-log_config', action='store_true', help='log config')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-wandb', action="store_true", help='set weights and biases')
    parser.add_argument('-adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    return parser


def load_model(model_config, device, distributed):
    model = get_image_classification_model(model_config, distributed)
    if model is None:
        repo_or_dir = model_config.get('repo_or_dir', None)
        model = get_model(model_config['key'], repo_or_dir, **model_config['kwargs'])
    src_ckpt_file_path = model_config.get('src_ckpt', None)
    load_ckpt(src_ckpt_file_path, model=model, strict=True)
    return model.to(device)


def train_one_epoch(training_box, device, epoch, log_freq):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets, supp_dict in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        sample_batch, targets = sample_batch.to(device), targets.to(device)
        loss = training_box.forward_process(sample_batch, targets, supp_dict)
        training_box.post_forward_process(loss=loss)
        batch_size = sample_batch.shape[0]
        metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError('The training loop was broken due to loss = {}'.format(loss))
    return loss


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
def evaluate(model, data_loader, device, device_ids, distributed, log_freq=1000, title=None, header='Test:'):
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


def train(teacher_model, student_model, dataset_dict, src_ckpt_file_path, dst_ckpt_file_path,
          device, device_ids, distributed, world_size, config, args):
    logger.info('Start training')
    train_config = config['train']
    lr_factor = world_size if distributed and args.adjust_lr else 1
    training_box = get_training_box(student_model, dataset_dict, train_config,
                                    device, device_ids, distributed, lr_factor) if teacher_model is None \
        else get_distillation_box(teacher_model, student_model, dataset_dict, train_config,
                                  device, device_ids, distributed, lr_factor)
    best_val_top1_accuracy = 0.0
    optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    if file_util.check_if_exists(src_ckpt_file_path):
        best_val_top1_accuracy, _ = load_ckpt(src_ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    log_freq = train_config['log_freq']
    student_config = config["models"]["student_model"]
    remote_save = student_config.get("remote_save",False)
    bucket_name = student_config.get("bucket_name",None)
    checkpoint_key = student_config.get("checkpoint_key",None)


    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    start_time = time.time()
    for epoch in range(args.start_epoch, training_box.num_epochs):
        training_box.pre_epoch_process(epoch=epoch)
        loss = train_one_epoch(training_box, device, epoch, log_freq)
        val_top1_accuracy = evaluate(student_model, training_box.val_data_loader, device, device_ids, distributed,
                                     log_freq=log_freq, header='Validation:')
        if val_top1_accuracy > best_val_top1_accuracy and is_main_process():
            accuracy_checkpoint_key = checkpoint_key
            logger.info('Best top-1 accuracy: {:.4f} -> {:.4f}'.format(best_val_top1_accuracy, val_top1_accuracy))
            logger.info('Updating ckpt at {}'.format(dst_ckpt_file_path))
            best_val_top1_accuracy = val_top1_accuracy
            save_ckpt(model=student_model_without_ddp, \
                    optimizer=optimizer, \
                    lr_scheduler=lr_scheduler, \
                    best_value=best_val_top1_accuracy, \
                    args=args, \
                    output_file_path=dst_ckpt_file_path)
            if remote_save:
                bi.upload_checkpoint(dst_ckpt_file_path, bucket_name, accuracy_checkpoint_key)


        training_box.post_epoch_process()
        if args.wandb:
            wandb.log({
                "val_top1_accuracy": val_top1_accuracy,
                "epoch": epoch,
                "training_loss": loss,
            })

    if distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    training_box.clean_modules()


def main(args):

    set_basic_log_config()
    log_file_path = args.run_log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    distributed, world_size, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    logger.info(args)
    if not args.disable_cudnn_benchmark:
        cudnn.benchmark = True

    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))


    import_dependencies(config.get('dependencies', None))
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: you asked for a GPU but none is available")
        device = torch.device("cpu")
    else:
        print(f"You asked for device {args.device} and that's what you'll get")
        device = torch.device(args.device)
    dataset_dict = config['datasets']
    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    teacher_model =\
        load_model(teacher_model_config, device, distributed) if teacher_model_config is not None else None
    student_model_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    src_ckpt_file_path = student_model_config.get('src_ckpt', None)
    dst_ckpt_file_path = student_model_config['dst_ckpt']

    student_model = load_model(student_model_config, device, distributed)
    ##WANDB INIT
    if args.wandb:
        run = wandb.init(
                project=args.project_name,  # Specify your project
                config={                        # Track hyperparameters and metadata
                    "run_log": args.run_log,
                    "device": args.device,
                    "teacher_model_config": teacher_model_config,
                    "student_model_config": student_model_config,
                },
            )


    if args.log_config:
        logger.info(config)

    if not args.test_only:
        train(teacher_model, student_model, dataset_dict, src_ckpt_file_path, dst_ckpt_file_path,
              device, device_ids, distributed, world_size, config, args)

    student_model_without_ddp =\
        student_model.module if module_util.check_if_wrapped(student_model) else student_model
    load_ckpt(dst_ckpt_file_path, model=student_model_without_ddp, strict=True)
    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                         test_data_loader_config, distributed)
    log_freq = test_config.get('log_freq', 1000)
    cudnn.benchmark = False
    cudnn.deterministic = True
    if not args.student_only and teacher_model is not None:
        evaluate(teacher_model, test_data_loader, device, device_ids, distributed, log_freq=log_freq,
                 title='[Teacher: {}]'.format(teacher_model_config['key']))
    evaluate(student_model, test_data_loader, device, device_ids, distributed, log_freq=log_freq,
             title='[Student: {}]'.format(student_model_config['key']))


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
