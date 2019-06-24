# !/bin/bash

cd ..
cd ..

# the folder saved the text data
TEXT_DATADIR='data/text_data/Processed_data_for_onmt'

# the folder to save the onmt-processed data
OUT_DIR='data/onmt_processed_data/full_dataset/'
mkdir -p ${OUT_DIR}

# the folde to save the process logs
LOG_DIR='logs/preprocess/'
mkdir -p ${LOG_DIR}

# "nohup" and "&" is used to run the code in the background
nohup /research/king3/wchen/Anaconda3/envs/py3.6_th0.4.1_cuda9.0/bin/python -u preprocess.py \
-train_src=${TEXT_DATADIR}/Training/word_kp20k_training_context_filtered.txt \
-train_tgt=${TEXT_DATADIR}/Training/word_kp20k_training_keyword_filtered.txt \
-train_key_indicators=${TEXT_DATADIR}/Training/word_kp20k_training_key_indicators_filtered.txt \
-train_retrieved_keys=${TEXT_DATADIR}/Training/word_kp20k_training_context_nstpws_sims_retrieved_keyphrases_filtered.txt \
-valid_src=${TEXT_DATADIR}/Validation/word_kp20k_validation_context_filtered.txt \
-valid_tgt=${TEXT_DATADIR}/Validation/word_kp20k_validation_keyword_filtered.txt \
-valid_key_indicators=${TEXT_DATADIR}/Validation/word_kp20k_validation_key_indicators_filtered.txt \
-valid_retrieved_keys=${TEXT_DATADIR}/Validation/word_kp20k_validation_context_nstpws_sims_retrieved_keyphrases_filtered.txt \
-save_data=${OUT_DIR}full_dataset \
-max_shard_size=65536000 \
-src_vocab_size=50000 \
-src_seq_length=400 \
-tgt_seq_length=6 \
-dynamic_dict \
-share_vocab > ${LOG_DIR}full_dataset_onmt_process_log.txt &