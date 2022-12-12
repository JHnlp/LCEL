#! /bin/sh

CUDA_VISIBLE_DEVICES=0 python litcovid_main.py \
  --data_dir=./data \
  --output_dir=./output/_multiclass_jnl_ext.training_augmentation.pubmedbert.asl_0_1_005 \
  --model_wrapper_name=pubmedbert \
  --tensorboard_comment=jnl_ext.training_augmentation.pubmedbert.asl_0_1_005 \
  --tokenizer_name=./model/Journal_Extension/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --model_name_or_path=./model/Journal_Extension/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/pytorch_model.bin \
  --config_name=./model/Journal_Extension/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/config_asl_0_1_005.json \
  --train_file=./data/new_BC7-LitCovid-Train-Data-Augmentation.csv \
  --dev_file=./data/new_BC7-LitCovid-Dev.csv \
  --test_file=./data/new_BC7-LitCovid-Test.csv \
  --log_file=./_log_litcovid_jnl_ext.training_augmentation.pubmedbert.asl_0_1_005.txt \
  --task_name=litcovid \
  --do_train \
  --eval_all_checkpoints \
  --do_lower_case \
  --overwrite_output_dir \
  --max_seq_length=512 \
  --per_gpu_train_batch_size=2 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=1e-5 \
  --num_train_epochs=3.0 \
  --logging_steps=50 \
  --save_steps=500 \
  --weight_decay=1e-5 \
  --warmup_proportion=0.05 \
  --max_step=20000 \
  --gradient_accumulation_steps=25
