# KG-KE-KR-M
The processed datasets and source code for the NAACL19 paper "An Integrated Approach for Keyphrase Generation via Exploring the Power of Retrieval and Extraction".
# Get the filtered KP20k training dataset
You can download the filtered **KP20k** training dataset [here](https://www.dropbox.com/s/kozr13nmw6cvb2q/kp20k_training_filtered.zip?dl=1). The statistics of the file are shown in the following table. Each empty (filtered) sample is stored as {"title": "", "keyword": "", "abstract": ""}.

Part | Number
--- | ---
Total | 530,802
Valid | 509,818
Empty | 20,984

The original training dataset from [Rui Meng](https://github.com/memray/seq2seq-keyphrase) contains 530809 data samples. We do the following filtering procedures:

1. We filter out the data samples with empty title or empty abstract. 7 samples are filtered and 530,802 samples are remained.
2. From the 530,802 samples, we filter out 20,984 samples and remain 509,818 samples:
    - 2.1 The samples without valid keyphrases that contain 1-6 tokens. 168 samples are filtered.
    - 2.2 The duplicated samples with the **KP20k** traning dataset itself, the **KP20k** validation dataset, the **KP20k** testing dataset, the **Inspec** testing dataset, the **Krapivin** testing dataset, the **NUS** testing dataset, and the **SemEval** testing dataset. 20,816 duplicated samples are filtered. We detect the duplications according to the following rules:
       - a. The Jaccard similarity between the corresponding non-stop-word sets of these two papers is larger or equal than 0.7.
         
         OR
       - b. The title of two papers are the same.


# Run our model
The source code will be available soon.
