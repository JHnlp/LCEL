"""
=======================================================================
 Copyright (c) 2019 PolyU CBS LLT Group. All Rights Reserved

@Author      :  Jinghang GU
@Contect     :  gujinghangnlp@gmail.com
@Time        :  2021/01/01
@Description :
=======================================================================
"""
import argparse
import json
import pprint
import logging
import os
import collections
import numpy as np
import pathlib
import csv
import pandas as pd
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import (WEIGHTS_NAME,
                          BertConfig, BertTokenizer,  # BertForSequenceClassification,
                          RobertaConfig, RobertaTokenizer,  # RobertaForSequenceClassification,
                          ElectraConfig, ElectraTokenizer,  # ElectraForSequenceClassification,
    # AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
    # XLMConfig, XLMForSequenceClassification, XLMTokenizer,
    # XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
    # GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer
                          )
from transformers import (AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup)
from biomul.utils.helper_functions import seed_everything, ModelEma
from biomul.processors.convert import convert_examples_to_features, processors
from biomul.evaluation.metrics import compute_metrics
from biomul.evaluation.biocreative_litcovid_eval import (validate, print_label_based_scores,
                                                         print_instance_based_scores)
from biomul.model.classifiers import (BertForSequenceClassification,
                                      RobertaForSequenceClassification,
                                      ElectraForSequenceClassification)

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)

MODEL_WRAPPER2TYPE = {
    'biobert': 'bert',
    'biobert_base': 'bert',
    'biobert_large': 'bert',
    'pubmedbert': 'bert',
    'covidbert': 'bert',
    'bioelectra': 'electra',
    'biom_electra': 'electra',
    'biomed_roberta': 'roberta',
    'specter': 'bert'
}

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # 'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    # 'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # 'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    # 'gpt2': (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'electra': (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer)
}


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        torch.distributed.barrier()

    processor = processors[task]()
    # Load data features from cache or dataset file
    if data_type == 'train':
        file_path = pathlib.Path(args.train_file)
        assert file_path.is_file(), '!!! Training file path error !!!'
    elif data_type == 'dev':
        file_path = pathlib.Path(args.dev_file)
        assert file_path.is_file(), '!!! Development file path error !!!'
    elif data_type == 'test':
        file_path = pathlib.Path(args.test_file)
        assert file_path.is_file(), '!!! Test file path error !!!'
    else:
        raise ValueError('!!! Specified wrong data type !!!')
    cached_features_file = os.path.join(args.data_dir,
                                        'cached_{}_{}_{}_{}'.format(
                                            args.model_wrapper_name, data_type,
                                            args.max_seq_length, file_path.stem))
    print()

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels(args.data_dir)

        examples = processor.get_examples(data_type=data_type, file_path=file_path)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                sequence_a_segment_id=0, sequence_b_segment_id=1,
                                                mask_padding_with_zero=True)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_guids = torch.tensor(np.array([int(f.guid) for f in features]), dtype=torch.long)  # load as integer
    all_input_ids = torch.tensor(np.array([f.input_indicators for f in features]), dtype=torch.long)
    all_input_mask = torch.tensor(np.array([f.attention_mask for f in features]), dtype=torch.long)
    all_token_type_ids = torch.tensor(np.array([f.token_type_indicators for f in features]), dtype=torch.long)
    all_label_ids = torch.tensor(np.array([f.labels for f in features]), dtype=torch.long)

    dataset = TensorDataset(all_guids, all_input_ids, all_input_mask, all_token_type_ids, all_label_ids)
    return dataset


def train(args, train_data, model, tokenizer, label_list, eval_data=None):
    """ Train the model """
    tb_writer = None
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(comment='-' + args.tensorboard_comment)

    # label_map = {label: i for i, label in enumerate(label_list)}
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)
    logger.info("Total training step %s", t_total)
    logger.info("Total training epochs %s", args.num_train_epochs)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    logger.info("Optimizer: %s", optimizer)
    logger.info("Scheduler: %s", scheduler)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    print()
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    scaler = GradScaler(enabled=args.use_amp)
    unscale_tag = False
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_loss = 1e8
    model.zero_grad()
    epoch_iter = trange(int(args.num_train_epochs), desc="<--- Epoch --->", disable=args.local_rank not in [-1, 0])
    # seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for _ in epoch_iter:
        with tqdm(train_dataloader, position=args.local_rank,
                  desc='==>Training Rank: {}, Batch size: {}'.format(
                      args.local_rank, args.per_gpu_train_batch_size)) as epoch_iterator:
            model.train()
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids': batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM and RoBERTa don't use segment_ids
                          'labels': batch[4].float()}
                batch_pmids = batch[0] if isinstance(batch[0], np.ndarray) \
                    else batch[0].detach().cpu().numpy()

                with autocast(enabled=args.use_amp):
                    outputs = model(**inputs)
                    loss, logits = outputs[:2]  # model outputs are always tuple in pytorch-transformers (see doc)

                prediction = torch.sigmoid(logits).detach().cpu().numpy()
                target = inputs['labels'].detach().cpu().numpy()  # candidate mesh descriptor mask for true labels

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()

                # Unscales the gradients of optimizer's assigned params in-place
                if not unscale_tag:
                    scaler.unscale_(optimizer)
                    unscale_tag = True

                # # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
                # # You may use the same value for max_norm here as you would without gradient scaling.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

                tr_loss += loss.item()
                if args.local_rank in [-1, 0]:
                    tb_writer.add_scalar('batch_loss', loss.item(), step)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    unscale_tag = False
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()  # set_to_none=True here can modestly improve performance
                    global_step += 1

                    if args.local_rank in [-1, 0]:
                        # tb_writer.add_scalar('stepped_lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('optimizer_lr', optimizer.param_groups[0]['lr'], global_step)

                        if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                            keep_info = collections.OrderedDict()
                            pred_binary = np.where(prediction > 0.5, 1, 0)
                            keep_info['All_target_sum'] = round(np.log2(target.sum() + args.epsilon), 6)
                            keep_info['All_bin_pred_sum'] = round(np.log2(pred_binary.sum() + args.epsilon), 6)
                            keep_info['All_logits_pred_sum'] = round(np.log2(prediction.sum() + args.epsilon), 6)
                            keep_info.update(compute_metrics(args.task_name, pred_binary, target))
                            for k, v in keep_info.items():
                                tb_writer.add_scalar(k, v, global_step)

                            average_loss = (tr_loss - logging_loss) / args.logging_steps
                            tb_writer.add_scalar('train_loss', average_loss, global_step)
                            logging_loss = tr_loss

                            if average_loss < best_loss:
                                best_loss = average_loss

                            logger.info("GPU Rank: %s, global_step: %s, train_loss: %1.5e, "
                                        "lr: %1.5e,  DocIDs: %s, Acc info: %s",
                                        args.local_rank, global_step, average_loss,
                                        optimizer.param_groups[0]['lr'],
                                        batch_pmids, keep_info)

                        if args.save_steps > 0 and global_step % args.save_steps == 0:
                            # Save model checkpoint
                            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            tokenizer.save_pretrained(save_directory=output_dir)

                            # evaluate
                            if args.local_rank == -1 and eval_data:  # Only evaluate when single GPU otherwise metrics may not average well
                                results = evaluate(args, eval_data, model, label_list,
                                                   prefix="do_eval_checkpoint_" + str(global_step))

                    # epoch_iterator(step, {'loss': loss.item(),
                    #             'lr': scheduler.get_lr()[0]})
                    epoch_iterator.set_postfix({'global step': '{}'.format(global_step),
                                                'batch_loss': '{0:1.5e}'.format(loss.item()),
                                                'lr': '{0:1.5e}'.format(optimizer.param_groups[0]['lr'])})
                    if args.max_steps > 0 and global_step > args.max_steps:
                        break
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, eval_data, model, label_list, prefix=""):
    """ There should be only one process runs the evaluation. """
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(eval_data, sampler=SequentialSampler(eval_data),
                                 batch_size=args.eval_batch_size)

    results = {}
    logger.info("***** Running evaluation on {} *****".format(prefix))
    logger.info("===>  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    all_pred_logits = None
    all_ground_truth = None
    all_pmids = None
    keep_info = collections.OrderedDict()
    with tqdm(eval_dataloader, desc="==>Evaluating, rank: {}".format(args.local_rank),
              position=args.local_rank + 1) as eval_iter:
        model.eval()
        for n_step, batch in enumerate(eval_iter, 1):
            with torch.no_grad():
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids': batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM and RoBERTa don't use segment_ids
                          'labels': batch[4].float()}  # convert to float for computing
                batch_pmids = batch[0] if isinstance(batch[0], np.ndarray) \
                    else batch[0].detach().cpu().numpy()

                outputs = model(**inputs)
                batch_loss, logits = outputs[:2]
                # logger.info("batch loss: %s", batch_loss)
                eval_loss += batch_loss.mean().item()

                batch_pred_logits = torch.sigmoid(logits).detach().cpu().numpy()
                batch_ground_truth = inputs['labels'].detach().cpu().numpy()

            if all_pred_logits is None:
                all_pred_logits = batch_pred_logits
                all_ground_truth = batch_ground_truth
                all_pmids = batch_pmids
            else:
                all_pred_logits = np.append(all_pred_logits, batch_pred_logits, axis=0)
                all_ground_truth = np.append(all_ground_truth, batch_ground_truth, axis=0)
                all_pmids = np.append(all_pmids, batch_pmids, axis=0)
            eval_iter.set_postfix({'eval step': '{}'.format(n_step)})

    final_binary_prediction = np.where(all_pred_logits > 0.5, 1, 0)
    keep_info['DocNum'] = len(all_pmids)
    keep_info['All_target_sum'] = round(np.log2(all_ground_truth.sum() + args.epsilon), 6)
    keep_info['All_bin_pred_sum'] = round(np.log2(final_binary_prediction.sum() + args.epsilon), 6)
    keep_info['All_logits_pred_sum'] = round(np.log2(all_pred_logits.sum() + args.epsilon), 6)
    metrics_res = compute_metrics(args.task_name, final_binary_prediction, all_ground_truth)
    keep_info.update(metrics_res)

    eval_loss = eval_loss / n_step
    results['eval_loss'] = eval_loss
    results.update(metrics_res)

    logger.info("Evaluation info: %s", keep_info)
    assert len(all_pmids) == len(final_binary_prediction) == len(all_ground_truth)

    columns = ['PMID'] + label_list
    pd_ground_truth = np.hstack([all_pmids.reshape(-1, 1), all_ground_truth]).astype(int)
    pd_ground_truth = pd.DataFrame(pd_ground_truth, columns=columns)
    pd_final_pred = np.hstack([all_pmids.reshape(-1, 1), final_binary_prediction]).astype(int)
    pd_final_pred = pd.DataFrame(pd_final_pred, columns=columns)

    #  official evaluation
    validate(pd_ground_truth, pd_final_pred, columns)
    lb_report = print_label_based_scores(pd_ground_truth, pd_final_pred, columns[1:])
    ex_report = print_instance_based_scores(pd_ground_truth, pd_final_pred, columns[1:])
    logger.info("lb_report: \n%s", lb_report)
    logger.info("ex_report: \n%s", ex_report)

    # output in csv format
    fn_prefix = pathlib.Path(args.dev_file).stem
    ground_truth_output_path = pathlib.Path(args.output_dir).joinpath(fn_prefix + '_' + prefix + '.gold.csv')
    pd_ground_truth.to_csv(ground_truth_output_path.absolute(), index=False)
    final_pred_output_path = pathlib.Path(args.output_dir).joinpath(fn_prefix + '_' + prefix + '.pred.csv')
    pd_final_pred.to_csv(final_pred_output_path.absolute(), index=False)

    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    return results


def parse_args():
    """
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
    python main.py --model_type=bert \
            --model_name_or_path=./model/bert/torch/cased_L-12_H-768_A-12 \
            --config_name=./model/bert/torch/cased_L-12_H-768_A-12/bert_config.json \
            --task_name=covid19 \
            --do_train \
            --do_eval \
            --do_lower_case \
            --data_dir=./data/Corpus \
            --max_seq_length=512 \
            --per_gpu_train_batch_size=16 \
            --per_gpu_eval_batch_size=16 \
            --learning_rate=2e-5 \
            --num_train_epochs=2.0 \
            --logging_steps=100 \
            --save_steps=100 \
            --output_dir=./output/covid19/ \
            --overwrite_output_dir

    # python distributed.py -bk nccl -im tcp://10.10.10.1:12345 -rn 0 -ws 2
    """
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--mesh_file", default=None, type=str, required=True,
    #                     help="The input MeSH terms file.")
    # parser.add_argument("--journal_file", default=None, type=str, required=True,
    #                     help="The input journal freq file.")
    parser.add_argument("--model_wrapper_name", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--tensorboard_comment", default=None, type=str, required=True,
                        help="The writer comment for tensorboard.")
    # parser.add_argument("--optimizer", default=None, type=str, required=True, help="Specify the optimizer.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run prediction on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--use_amp', action='store_true',
                        help="Whether to use Automatic Mixed Precision instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--train_file', type=str, default=None, help="The training file path.")
    parser.add_argument('--dev_file', type=str, default=None, help="The development file path.")
    parser.add_argument('--test_file', type=str, default=None, help="The test file path.")
    parser.add_argument('--log_file', type=str, default=None, help="The log file path.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # pprint.pprint(args.__dict__)  # print args parameters

    # rebase root directory
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    print('*' * 50)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.INFO)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    file_hdl = logging.FileHandler(args.log_file, mode='w', encoding='UTF-8')
    file_hdl.setLevel(logging.INFO)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console_hdl = logging.StreamHandler()
    console_hdl.setLevel(logging.INFO)
    # 为logger对象添加句柄
    logger.addHandler(file_hdl)
    logger.addHandler(console_hdl)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, automatic mixed precision: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.use_amp)

    # Set seed
    # seed_everything(args.seed)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_wrapper_name = args.model_wrapper_name.lower()
    args.model_type = MODEL_WRAPPER2TYPE[args.model_wrapper_name]
    # args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    logger.info("==> Task Model Type: {}".format(model_class))

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    # config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #                                           do_lower_case=args.do_lower_case)
    # model_2 = AutoModel.from_pretrained(args.model_name_or_path,
    #                                   from_tf=bool('.ckpt' in args.model_name_or_path),
    #                                   config=config)
    #
    # add new mark tokens
    num_of_words_to_add = tokenizer.add_tokens(['covid', 'coronavirus', 'sars'])
    num_of_special_to_add = tokenizer.add_special_tokens(
        {"additional_special_tokens": ['<TAG>', '<KEYWORD>', '<MESH>']})
    if num_of_special_to_add:  # add new embeddings or just pass
        model.resize_token_embeddings(tokenizer.vocab_size + num_of_words_to_add + num_of_special_to_add)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    print('*' * 50)
    logger.info("On GPU {}, Training/evaluation parameters {}".format(args.local_rank, args))

    train_dataset, eval_dataset = None, None
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, label_list, eval_data=eval_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        print('*' * 50)

        # Saving the last & best practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            if global_step % args.save_steps != 0:
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                logger.info("Saving the last model checkpoint to %s", output_dir)
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                # Good practice: save your training arguments together with the trained model
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                # # Load a trained model and vocabulary that you have fine-tuned
                # model = model_class.from_pretrained(output_dir)
                # tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
                # model.to(args.device)

    print('*' * 50)
    # Evaluation
    if args.do_eval:
        print(' ==> evaluating {}'.format(args.dev_file))
        eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer,
                                               data_type='dev') if not eval_dataset else eval_dataset
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = sorted(pathlib.Path(args.output_dir).glob('**/' + WEIGHTS_NAME))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

        if args.local_rank == -1:
            checkpoints_for_local_rank = checkpoints
            logger.info("Single mode evaluating, Local Rank -1.")
        else:
            world_size = torch.distributed.get_world_size()
            logger.info("Multi mode evaluating, World Size: %s, Local Rank: %s",
                        world_size, args.local_rank)

            checkpoints_for_local_rank = []
            for idx, checkpoint in enumerate(checkpoints):
                if idx % world_size == args.local_rank:
                    checkpoints_for_local_rank.append(checkpoint)

        logger.info("Rank %s GPU is evaluating the following checkpoints: %s",
                    args.local_rank, checkpoints_for_local_rank)

        eval_results = collections.OrderedDict()
        columns = set()
        for checkpoint in checkpoints_for_local_rank:
            global_step = ''
            if 'checkpoint' in checkpoint.parent.name \
                    and len(checkpoint.parent.name.split('-')) > 1:
                global_step = checkpoint.parent.name.split('-')[-1]

            config = config_class.from_pretrained(checkpoint.parent.joinpath('config.json'))
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args, eval_dataset, model, label_list, prefix=global_step)
            columns.update(result.keys())
            eval_results[str(checkpoint.parent.name)] = result

            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
        columns = sorted(columns)
        write_rank_id = 'LocalRank_{}_'.format(args.local_rank) if args.local_rank >= 0 else 'Single_'
        with pathlib.Path(args.output_dir).joinpath(
                '{}_eval_records.___.{}.___.csv'.format(write_rank_id, pathlib.Path(args.dev_file).stem)
        ).open('w', encoding='utf8') as outf:
            tsv_w = csv.writer(outf, delimiter=',')
            tsv_w.writerow(['Entry'] + columns)
            for key, values in eval_results.items():
                record = [key] + [values[col] for col in columns]
                tsv_w.writerow(record)


if __name__ == "__main__":
    main()

    pass
