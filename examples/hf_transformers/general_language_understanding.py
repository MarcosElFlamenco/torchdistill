"""
Example code to fine-tuning the Transformer models for sequence classification on the GLUE benchmark
at https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py
modified to collaborate with torchdistill.

Original copyright of The HuggingFace Inc. code below, modifications by Yoshitomo Matsubara, Copyright 2021.
"""
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import time

import datasets
import evaluate as hf_evaluate
import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from torch.backends import cudnn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, \
    PretrainedConfig, default_data_collator

from custom.optim import customize_lr_config
from torchdistill.common import file_util, yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, setup_for_distributed, set_seed, import_dependencies
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import get_training_box
from torchdistill.datasets import util
from torchdistill.datasets.registry import register_collate_func
from torchdistill.misc.log import set_basic_log_config, setup_log_file, SmoothedValue, MetricLogger

logger = def_logger.getChild(__name__)

GLUE_TASK2KEYS = {
    'ax': ('premise', 'hypothesis'),
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
}

register_collate_func(default_data_collator)


def get_argparser():
    parser = argparse.ArgumentParser(description='Knowledge distillation for text classification models')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--run_log', help='log file path')
    parser.add_argument('--task_name', type=str, default=None, help='name of the glue task to train on.')
    parser.add_argument('--private_output', help='output dir path for private dataset(s)')
    parser.add_argument('--seed', type=int, default=None, help='a seed for reproducible training')
    parser.add_argument('-disable_cudnn_benchmark', action='store_true', help='disable torch.backend.cudnn.benchmark')
    parser.add_argument('-test_only', action='store_true', help='only test the models')
    parser.add_argument('-student_only', action='store_true', help='test the student model only')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('-adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    return parser


def load_tokenizer_and_model(model_config, task_name, prioritizes_dst_ckpt=False):
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_config = model_config['config_kwargs']
    config = AutoConfig.from_pretrained(**config_config, finetuning_task=task_name)
    tokenizer_config = model_config['tokenizer_kwargs']
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
    model_kwargs = model_config['model_kwargs']
    if prioritizes_dst_ckpt and file_util.check_if_exists(model_config.get('dst_ckpt', None)):
        model_kwargs['pretrained_model_name_or_path'] = model_config['dst_ckpt']
    elif file_util.check_if_exists(model_config.get('src_ckpt', None)):
        model_kwargs['pretrained_model_name_or_path'] = model_config['src_ckpt']
    model = AutoModelForSequenceClassification.from_pretrained(config=config, **model_kwargs)
    return tokenizer, model


def load_raw_glue_datasets_and_misc(task_name, train_file_path=None, valid_file_path=None, base_split_name='train'):
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset('glue', task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if train_file_path is not None:
            data_files['train'] = train_file_path
        if valid_file_path is not None:
            data_files['validation'] = valid_file_path

        extension = (train_file_path if train_file_path is not None else valid_file_path).split('.')[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    label_names = None
    if task_name is not None:
        is_regression = task_name == 'stsb'
        if not is_regression:
            label_names = raw_datasets[base_split_name].features['label'].names
            num_labels = len(label_names)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets[base_split_name].features['label'].dtype in ['float32', 'float64']
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_names = raw_datasets[base_split_name].unique('label')
            label_names.sort()  # Let's sort it for determinism
            num_labels = len(label_names)
    return raw_datasets, num_labels, label_names, is_regression


def preprocess_glue_datasets(task_name, raw_datasets, num_labels, label_names, is_regression,
                             pad_to_max_length, max_length, tokenizer, model, base_split_name='train'):
    # Preprocessing the datasets
    if task_name is not None:
        sentence1_key, sentence2_key = GLUE_TASK2KEYS[task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets[base_split_name].column_names if name != 'label']
        if 'sentence1' in non_label_column_names and 'sentence2' in non_label_column_names:
            sentence1_key, sentence2_key = 'sentence1', 'sentence2'
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_names)):
            logger.info(
                f'The configuration of the model provided the following label correspondence: {label_name_to_id}. '
                'Using it!'
            )
            label_to_id = {i: label_name_to_id[label_names[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f'model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_names))}.'
                '\nIgnoring the model labels as a result.',
            )
    elif task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_names)}

    padding = 'max_length' if pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        if 'label' in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result['labels'] = [label_to_id[l] for l in examples['label']]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result['labels'] = examples['label']
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets[base_split_name].column_names
    )
    return processed_datasets


def get_all_datasets(datasets_config, task_name, student_tokenizer, student_model):
    dataset_dict = dict()
    label_names_dict = dict()
    is_regression = None
    for dataset_name in datasets_config.keys():
        dataset_config = datasets_config[dataset_name]
        raw_data_kwargs = dataset_config['raw_data_kwargs']
        base_split_name = dataset_config.get('base_split_name', 'train')
        sub_task_name = dataset_config.get('name', task_name)
        raw_datasets, num_labels, label_names, is_regression = \
            load_raw_glue_datasets_and_misc(sub_task_name, base_split_name=base_split_name, **raw_data_kwargs)
        pad_to_max_length = dataset_config.get('pad_to_max_length', False)
        max_length = dataset_config.get('max_length', 128)
        sub_dataset_dict = \
            preprocess_glue_datasets(sub_task_name, raw_datasets, num_labels, label_names, is_regression,
                                     pad_to_max_length, max_length, student_tokenizer, student_model, base_split_name)
        for split_name, dataset_id in dataset_config['dataset_id_map'].items():
            dataset_dict[dataset_id] = sub_dataset_dict[split_name]
        label_names_dict[sub_task_name] = label_names
    return dataset_dict, label_names_dict, is_regression


def get_metrics(task_name):
    metric = hf_evaluate.load('glue', task_name)
    return metric


def train_one_epoch(training_box, epoch, log_freq):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('sample/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()
        loss = training_box.forward_process(sample_batch, targets=None, supp_dict=None)
        training_box.post_forward_process(loss=loss)
        batch_size = len(sample_batch)
        metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['sample/s'].update(batch_size / (time.time() - start_time))


@torch.inference_mode()
def evaluate(model, data_loader, metric, is_regression, accelerator, title=None, header='Test: '):
    if title is not None:
        logger.info(title)

    model.eval()
    for batch in data_loader:
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch['labels']),
        )

    eval_dict = metric.compute()
    eval_desc = header + ', '.join([f'{key} = {eval_dict[key]}' for key in sorted(eval_dict.keys())])
    logger.info(eval_desc)
    return eval_dict


def train(teacher_model, student_model, dataset_dict, is_regression, dst_ckpt_dir_path, metric,
          device, device_ids, distributed, config, args, accelerator):
    logger.info('Start training')
    train_config = config['train']
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    training_box = get_training_box(student_model, dataset_dict, train_config,
                                    device, device_ids, distributed, lr_factor, accelerator) if teacher_model is None \
        else get_distillation_box(teacher_model, student_model, dataset_dict, train_config,
                                  device, device_ids, distributed, lr_factor, accelerator)
    # Only show the progress bar once on each machine.
    log_freq = train_config['log_freq']
    best_val_number = 0.0
    for epoch in range(training_box.num_epochs):
        training_box.pre_epoch_process(epoch=epoch)
        train_one_epoch(training_box, epoch, log_freq)
        val_dict = evaluate(student_model, training_box.val_data_loader, metric, is_regression,
                            accelerator, header='Validation: ')
        val_value = sum(val_dict.values())
        if val_value > best_val_number:
            logger.info('Updating ckpt at {}'.format(dst_ckpt_dir_path))
            best_val_number = val_value
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(student_model)
            unwrapped_model.save_pretrained(dst_ckpt_dir_path, save_function=accelerator.save)
        training_box.post_epoch_process()


@torch.inference_mode()
def predict_private(model, dataset_dict, label_names_dict, is_regression, accelerator,
                    private_configs, private_output_dir_path):
    logger.info('Start prediction for private dataset(s)')
    model.eval()
    for private_config in private_configs:
        # Dataset
        private_data_loader_config = private_config['private_data_loader']
        private_dataset_id = private_data_loader_config['dataset_id']
        private_dataset = dataset_dict[private_dataset_id]
        label_names = label_names_dict[private_data_loader_config['task_name']]
        logger.info('{}: {} samples'.format(private_dataset_id, len(private_dataset)))

        # Dataloader
        private_data_loader = util.build_data_loader(private_dataset, private_data_loader_config, False)
        private_data_loader = accelerator.prepare(private_data_loader)

        # Prediction
        private_output_file_path = os.path.join(private_output_dir_path, private_config['pred_output'])
        file_util.make_parent_dirs(private_output_file_path)
        np_preds = None
        for batch in private_data_loader:
            batch.pop('labels')
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions = predictions.detach().cpu().numpy()
            np_preds = predictions if np_preds is None else np.append(np_preds, predictions, axis=0)

        df_output = pd.DataFrame({'prediction': np_preds})
        # Map prediction index to label name
        if not is_regression and private_config.get('idx2str', True):
            df_output.prediction = df_output.prediction.apply(lambda pred_idx: label_names[pred_idx])
        df_output.to_csv(private_output_file_path, sep='\t', index=True, index_label='index')


def main(args):
    set_basic_log_config()
    log_file_path = args.run_log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    world_size = args.world_size
    logger.info(args)
    if not args.disable_cudnn_benchmark:
        cudnn.benchmark = True

    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    import_dependencies(config.get('dependencies', None))

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    distributed = accelerator.state.distributed_type == DistributedType.MULTI_GPU
    device_ids = [accelerator.device.index]
    if distributed:
        setup_for_distributed(is_main_process())

    logger.info(accelerator.state)
    device = accelerator.device

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Load pretrained model and tokenizer
    task_name = args.task_name
    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    teacher_tokenizer, teacher_model = (None, None) if teacher_model_config is None \
        else load_tokenizer_and_model(teacher_model_config, task_name, True)
    student_model_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    student_tokenizer, student_model = load_tokenizer_and_model(student_model_config, task_name, False)
    dst_ckpt_dir_path = student_model_config['dst_ckpt']
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    dataset_dict, label_names_dict, is_regression = \
        get_all_datasets(config['datasets'], task_name, student_tokenizer, student_model)

    # Update config with dataset size len(data_loader)
    customize_lr_config(config, dataset_dict, world_size)

    # register collate function
    register_collate_func(DataCollatorWithPadding(student_tokenizer,
                                                  pad_to_multiple_of=(8 if accelerator.use_fp16 else None)))

    # Get the metric function
    metric = get_metrics(task_name)

    if not args.test_only:
        train(teacher_model, student_model, dataset_dict, is_regression, dst_ckpt_dir_path, metric,
              device, device_ids, distributed, config, args, accelerator)
        student_tokenizer.save_pretrained(dst_ckpt_dir_path)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, distributed)
    test_data_loader = accelerator.prepare(test_data_loader)
    cudnn.benchmark = False
    cudnn.deterministic = True
    if not args.student_only and teacher_model is not None:
        teacher_model = teacher_model.to(accelerator.device)
        evaluate(teacher_model, test_data_loader, metric, is_regression, accelerator,
                 title='[Teacher: {}]'.format(teacher_model_config['key']))

    # Reload the best checkpoint based on validation result
    student_tokenizer, student_model = load_tokenizer_and_model(student_model_config, task_name, True)
    student_model = accelerator.prepare(student_model)
    evaluate(student_model, test_data_loader, metric, is_regression, accelerator,
             title='[Student: {}]'.format(student_model_config['key']))

    # Output prediction for private dataset(s) if both the config and output dir path are given
    private_configs = config.get('private', None)
    private_output_dir_path = args.private_output
    if private_configs is not None and private_output_dir_path is not None and is_main_process():
        predict_private(student_model, dataset_dict, label_names_dict, is_regression, accelerator,
                        private_configs, private_output_dir_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
