#!/bin/bash
#SBATCH --job-name=translate_full
#SBATCH --output=/research/king3/wchen/Code4Git/KG-KE-KR-M/KG-KE-KR/logs/translate/seed3435_full_kg_ke_kr_m_debug/translate_full_log.txt
#SBATCH --gres=gpu:1
#SBATCH -p gpu_24h
#SBATCH -w gpu26

# The above is the slurm configuration. The training log will be saved in the "--output" file. 
# Please change to your own log folder when run the code using a slurm server. 

# If you do not use a slurm server, you may need to set CUDA_VISIBLE_DEVICES to allocate an available GPU in the machine.
#export CUDA_VISIBLE_DEVICES=1
GPUID=0

cd ..
cd ..

MODEL="seed3435_full_kg_ke_kr"
MODEL_DIR="saved_models/end2end/${MODEL}/"

# the model file is "seed3435_full_kg_ke_kr_m_genPPL_9.704_aveMLoss_7.517_aveSelLoss_0.5045_aveIncLoss_0.000_selF1_0.588_genAcc_55.29_step_108000.pt"
# we remove ".pt" here to provide convenience for naming other log files
saved_model="seed3435_full_kg_ke_kr_m_genPPL_9.704_aveMLoss_7.517_aveSelLoss_0.5045_aveIncLoss_0.000_selF1_0.588_genAcc_55.29_step_108000"

LOG_DIR="logs/translate/${MODEL}/"
mkdir -p ${LOG_DIR}

beam_size=200
max_length=6
bs=1

Testing_DIR="data/text_data/Processed_data_for_onmt/Testing"

for DATASET in "inspec" "krapivin" "nus" "semeval"
do
  /research/king3/wchen/Anaconda3/envs/py3.6_th0.4.1_cuda9.0/bin/python translate.py \
  -model="${MODEL_DIR}${saved_model}.pt" \
  -output="${LOG_DIR}${MODEL}_${DATASET}.out" \
  -scores_output="${LOG_DIR}${MODEL}_${DATASET}_gen_scores.out" \
  -sel_probs_output="${LOG_DIR}${MODEL}_${DATASET}_sel_probs.out" \
  -src=${Testing_DIR}/word_${DATASET}_testing_context.txt \
  -retrieved_keys=${Testing_DIR}/word_${DATASET}_testing_context_nstpws_sims_retrieved_keyphrases_filtered.txt \
  -key_indicators=${Testing_DIR}/word_${DATASET}_testing_key_indicators.txt \
  -kpg_tgt=${Testing_DIR}/word_${DATASET}_testing_keyword.txt \
  -kpg_context=${Testing_DIR}/word_${DATASET}_testing_context.txt \
  -beam_size=${beam_size} \
  -max_length=${max_length} \
  -n_best=${beam_size} \
  -batch_size=${bs} \
  -gpu=${GPUID} \
  -single_word_maxnum=1 > "${LOG_DIR}${MODEL}_${DATASET}_log.txt"
done

DATASET="kp20k"
/research/king3/wchen/Anaconda3/envs/py3.6_th0.4.1_cuda9.0/bin/python translate.py \
-model="${MODEL_DIR}${saved_model}.pt" \
-output="${LOG_DIR}${MODEL}_${DATASET}.out" \
-scores_output="${LOG_DIR}${MODEL}_${DATASET}_gen_scores.out" \
-sel_probs_output="${LOG_DIR}${MODEL}_${DATASET}_sel_probs.out" \
-src=${Testing_DIR}/word_${DATASET}_testing_context.txt \
-retrieved_keys=${Testing_DIR}/word_${DATASET}_testing_context_nstpws_sims_retrieved_keyphrases_filtered.txt \
-key_indicators=${Testing_DIR}/word_${DATASET}_testing_key_indicators.txt \
-kpg_tgt=${Testing_DIR}/word_${DATASET}_testing_keyword.txt \
-kpg_context=${Testing_DIR}/word_${DATASET}_testing_context.txt \
-beam_size=${beam_size} \
-max_length=${max_length} \
-n_best=${beam_size} \
-batch_size=${bs} \
-gpu=${GPUID} \
-single_word_maxnum=-1 > "${LOG_DIR}${MODEL}_${DATASET}_log.txt"