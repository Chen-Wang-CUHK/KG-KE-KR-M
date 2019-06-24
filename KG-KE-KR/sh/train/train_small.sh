#!/bin/bash
#SBATCH --job-name=train_small
#SBATCH --output=/research/king3/wchen/Code4Git/KG-KE-KR-M/KG-KE-KR/logs/train/train_small_log.txt
#SBATCH --gres=gpu:1
#SBATCH -p gpu_24h
#SBATCH -w gpu23

# The above is the slurm configuration. The training log will be saved in the "--output" file.
# Please change to your own log folder when run the code using a slurm server.

# If you do not use a slurm server, you may need to set CUDA_VISIBLE_DEVICES to allocate a GPU
#export CUDA_VISIBLE_DEVICES=1
GPUID=0

# 343, 3435, 34350 are the seeds to run the experiments on the full dataset in the paper
SEED=3435
MODEL_TYPE="end2end"
MODEL_NAME="seed${SEED}_small_kg_ke_kr"

cd ..
cd ..

SAVED_DIR="saved_models/${MODEL_TYPE}/${MODEL_NAME}/"
mkdir -p ${SAVED_DIR}

/research/king3/wchen/Anaconda3/envs/py3.6_th0.4.1_cuda9.0/bin/python train.py \
-save_model=${SAVED_DIR}${MODEL_NAME} \
-data=data/onmt_processed_data/small_dataset/small_dataset \
-vocab=data/onmt_processed_data/small_dataset/small_dataset \
-model_type=text \
-share_embeddings \
-key_model=key_end2end \
-only_rescale_copy \
-use_retrieved_keys \
-e2e_type=share_enc_sel \
-sel_train_ratio=1.0 \
-sel_classifier=complex_Nallapati \
-global_attention=general \
-copy_attn \
-reuse_copy_attn \
-word_vec_size=100 \
-dropout=0.1 \
-encoder_type=brnn \
-enc_layers=1 \
-dec_layers=1 \
-rnn_size=300 \
-rnn_type=GRU \
-train_steps=3500 \
-save_checkpoint_steps=100 \
-valid_steps=100 \
-report_every=50 \
-batch_size=64 \
-valid_batch_size=64 \
-seed=${SEED} \
-pos_weight=9.0 \
-sel_lambda=1.0 \
-sel_normalize_by_length \
-gen_lambda=1.0 \
-incons_lambda=0.0 \
-optim=adam \
-max_grad_norm=1 \
-learning_rate=0.001 \
-learning_rate_decay=0.5 \
-gpuid=${GPUID}