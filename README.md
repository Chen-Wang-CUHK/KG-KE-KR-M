# KG-KE-KR-M
The processed datasets and source code for the NAACL19 paper "[An Integrated Approach for Keyphrase Generation via Exploring the Power of Retrieval and Extraction](https://arxiv.org/pdf/1904.03454.pdf)". The code is based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
# Dependencies
- python 3.6.6
- pytorch 0.4.1 (CUDA9.0)
- torchtext 0.3.1

The full dependencies are listed in `requirements.txt`.

# Get the filtered raw KP20k training dataset
You can download the filtered raw **KP20k** training dataset [here](https://www.dropbox.com/s/kozr13nmw6cvb2q/kp20k_training_filtered.zip?dl=1). The statistics of the file are shown in the following table. Each empty (filtered) sample is stored as {"title": "", "keyword": "", "abstract": ""}.

Part | Number
--- | ---
Total | 530,802
Valid | 509,818
Empty | 20,984

The original training dataset from [Rui Meng](https://github.com/memray/seq2seq-keyphrase) contains `530,809` data samples. We do the following filtering procedures:

1. We filter out the data samples with empty title or empty abstract. `7` samples are filtered and `530,802` samples are remained.
2. From the `530,802` samples, we filter out `20,984` samples and remain `509,818` samples:
    - 2.1 The samples without valid keyphrases that contain 1-6 tokens. `168` samples are filtered.
    - 2.2 The duplicated samples with the **KP20k** traning dataset itself, the **KP20k** validation dataset, the **KP20k** testing dataset, the **Inspec** testing dataset, the **Krapivin** testing dataset, the **NUS** testing dataset, and the **SemEval** testing dataset. `20,816` duplicated samples are filtered. We regard two samples (papers) are duplicated when either condition a or condiation b is satisfied:
       - a. The Jaccard similarity between the corresponding non-stop-word sets of these two papers is larger or equal than 0.7.
       - b. The title of two papers are the same.

Finally, we obtain `509,818` valid data samples.
# Data preprocess
1. We lowercase, tokenize (use [stanfordcorenlp](https://github.com/Lynten/stanford-corenlp)), and replace the digit with "\<digit\>" token for all the text. You can download the processed data [here](https://www.dropbox.com/s/lgeza7owhn9dwtu/Processed_data_for_onmt.zip?dl=1).
   - The `*_context_filtered.txt` files and `*_context.txt` files store the context of each paper (i.e. the title + ". \<eos\>" + the abstract).
   - The `Training/word_kp20k_training_keyword_filtered.txt` and `Validation/word_kp20k_validation_keyword_filtered.txt` store the keyphrases of the training and validation datasets respectively. Each line is a keyphrase.
   - The `*_key_indicators_filtered.txt` files store the keyword indicators for each context token. `I` (`O`) means the corresponding context token is (not) a keyword. A context token is regarded as a keyword if (1) it is a non-stop-word and (2) it is one token of one of the gold keyphrases of this paper. 
   - The `*_context_nstpws_sims_retrieved_keyphrases_filtered.txt` files store the concatenated retrieved keyphrases of the top 3 similar papers. The retrieved keyphrases are split by a `<eos>` token. We utilize Jaccard similarity score between the non-stop-word sets of the two papers as the similarity score. For all the training, validation, and testing datasets, we use the filtered **KP20k** training dataset as the retrieval corpus.
   - The `Testing\*_context_nstpws_sims_retrieved_scores_filtered.txt` files store the retrival score of each retrieved keyphrase, which is the corresponding Jaccard similarity score between the retrieved paper and the original paper.
   - The `Testing\*_testing_keyword.txt` files store the keyphrases of each testing paper. Each line contains all the keyphrases of a testing paper, which are split by a `;` token.
# Run our model
The source code will be available soon.
# Citation
You can cite our paper by:
```
@inproceedings{chen-etal-2019-integrated,
    title = "An Integrated Approach for Keyphrase Generation via Exploring the Power of Retrieval and Extraction",
    author = "Chen, Wang  and
      Chan, Hou Pong  and
      Li, Piji  and
      Bing, Lidong  and
      King, Irwin",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1292",
    pages = "2846--2856",
}
```
