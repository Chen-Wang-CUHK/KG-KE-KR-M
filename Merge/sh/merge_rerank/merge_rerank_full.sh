#!/bin/bash
#SBATCH --job-name=merge_rerank_full
#SBATCH --output=/research/king3/wchen/Code4Git/KG-KE-KR-M/Merge/logs/seed3435_full_kg_ke_kr_merge_rerank_log.txt
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

MG_KEY="merge_all"

DATA_DIR="../KG-KE-KR/data/text_data/Processed_data_for_onmt/Testing/"

IN_LOG_DIR="../KG-KE-KR/logs/translate/${MODEL}"
OUT_LOG_DIR="logs/translate/${MODEL}"
mkdir -p ${OUT_LOG_DIR}

for DATASET in 'inspec' 'krapivin' 'nus' 'semeval'
do
  /research/king3/wchen/Anaconda3/envs/py3.6_th0.4.1_cuda9.0/bin/python merge_rerank.py \
  -model=saved_models/seed3435_reranker.pt \
  -src=${DATA_DIR}/word_${DATASET}_testing_context.txt \
  -tgt=${IN_LOG_DIR}/${MODEL}_${DATASET}.out \
  -gen_scores=${IN_LOG_DIR}/${MODEL}_${DATASET}_gen_scores.out \
  -retrieved_keys=${DATA_DIR}/word_${DATASET}_testing_context_nstpws_sims_retrieved_keyphrases_filtered.txt \
  -retrieved_scores=${DATA_DIR}/word_${DATASET}_testing_context_nstpws_sims_retrieved_scores_filtered.txt \
  -sel_probs=${IN_LOG_DIR}/${MODEL}_${DATASET}_sel_probs.out \
  -merge_ex_keys \
  -merge_rk_keys \
  -merge_with_stemmer \
  -reranked_scores_output=${OUT_LOG_DIR}/${MODEL}_${DATASET}_${MG_KEY}_merged_reranked_scores.out \
  -output=${OUT_LOG_DIR}/${MODEL}_${DATASET}_${MG_KEY}_merged_reranked.out \
  -sel_keys_output=${OUT_LOG_DIR}/${MODEL}_${DATASET}_sel_keys.out \
  -gpu=${GPUID} \
  -kpg_context=${DATA_DIR}/word_${DATASET}_testing_context.txt \
  -kpg_tgt=${DATA_DIR}/word_${DATASET}_testing_keyword.txt \
  -match_method=word_match \
  -filter_dot_comma_unk=True \
  -single_word_maxnum=1 > ${OUT_LOG_DIR}/${MODEL}_${DATASET}_${MG_KEY}_merged_reranked_log.txt
done

DATASET='kp20k'
/research/king3/wchen/Anaconda3/envs/py3.6_th0.4.1_cuda9.0/bin/python merge_rerank.py \
-model=saved_models/seed3435_reranker.pt \
-src=${DATA_DIR}/word_${DATASET}_testing_context.txt \
-tgt=${IN_LOG_DIR}/${MODEL}_${DATASET}.out \
-gen_scores=${IN_LOG_DIR}/${MODEL}_${DATASET}_gen_scores.out \
-retrieved_keys=${DATA_DIR}/word_${DATASET}_testing_context_nstpws_sims_retrieved_keyphrases_filtered.txt \
-retrieved_scores=${DATA_DIR}/word_${DATASET}_testing_context_nstpws_sims_retrieved_scores_filtered.txt \
-sel_probs=${IN_LOG_DIR}/${MODEL}_${DATASET}_sel_probs.out \
-merge_ex_keys \
-merge_rk_keys \
-merge_with_stemmer \
-reranked_scores_output=${OUT_LOG_DIR}/${MODEL}_${DATASET}_${MG_KEY}_merged_reranked_scores.out \
-output=${OUT_LOG_DIR}/${MODEL}_${DATASET}_${MG_KEY}_merged_reranked.out \
-sel_keys_output=${OUT_LOG_DIR}/${MODEL}_${DATASET}_sel_keys.out \
-gpu=${GPUID} \
-kpg_context=${DATA_DIR}/word_${DATASET}_testing_context.txt \
-kpg_tgt=${DATA_DIR}/word_${DATASET}_testing_keyword.txt \
-match_method=word_match \
-filter_dot_comma_unk=True \
-single_word_maxnum=-1 > ${OUT_LOG_DIR}/${MODEL}_${DATASET}_${MG_KEY}_merged_reranked_log.txt