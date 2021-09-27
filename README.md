# Truth Detection using Natural Language Processing

#### Name: Sergiu-Ionut Craioveanu
#### Institution: University POLITEHNICA of Bucharest, Faculty of Automatic Control and Computer Science
#### Generation: 2017-2021

This repo contains all the code and data behind my BSc Thesis, split into intuitively named folders. Most of the code is within interactive Python notebooks (ipynb), so it should be easy to follow.

Below, I also present a few ways of reconstructing my local environment, where necessary.

# Introduction
This project sets out to answer two main questions:

- How does truthful language vary from misleading language?
- To what extent can we leverage Machine Learning or Deep Learning to correctly predict affirmations?

I devise my own case study based on [Politifact](https://www.politifact.com/truth-o-meter/) which is a site designed to verify political claims in USA, led by top people in multi-disciplinary fields. 
I parse the site and create my own database, leverage SpaCy to uncover predominant features in affirmations based on veracity, and formulate a problem of Truth/Lie classification. 
I experiment with various ML/AutoML/DL models and manage to correctly classify 2/3 of affirmations based on text alone.

# Code description

The code is split into stages of progression. Each folder contains notebooks for various tasks. The "Data" folder contains all generated data, alongside csv files with results.
```
📦1. Data Manipulation
 ┣ 📜1_a_PolitifactDataWebScraping.ipynb
 ┣ 📜1_b_PolitifactCreateNewVariants.ipynb
 ┣ 📜1_c_PolitifactCreateTrainTestsSplits.ipynb
 ┣ 📜1_d_PolitifactDataAugmentation.ipynb
 ┗ 📜1_e_Clean_Data_NER.ipynb
```
 The "Data Manipulation" folder contains the website parsing notebooks, alongside the notebooks I used for cleaning the data and creating potential Train/Test splits. Also used to create multiple variants of the dataset.
```
 📦2. Part of Speech Tagging & Named Entity Recognition
 ┣ 📜2_a_FeatureEngineering_POS_Stats.ipynb
 ┗ 📜2_b_AdvancedPOS_NER_SD_WC.ipynb
```
 The "Part of Speech Tagging & Named Entity Recognition" contains the process of creating separate columns based on features found within the text. They are later used to create statistics based on Entities, Parts of Speech and Syntactic Dependencies.
```
📦3. Machine Learning  & AutoML
 ┣ 📜3_a_ML_LR_MNB_TFIDF.ipynb
 ┗ 📜3_b_AutoKeras_Truth_Detection_Text_Classification.ipynb
```
 The Machine Learning aspect of the above folder attempts the classification of affirmations using Logistic Regression and Multinomial Naive Bayes models, with different word represenation models (Bag of Words, N-grams), with varying sentence normalizations (partial/full).

 The AutoML step leverages AutoKeras in an attempt to construct a text classification model architecture that best fits the need of the task, automatically, through various unsupervised trials.
```
 📦4. Deep Learning
 ┣ 📂politifact_binarized_augmented
 ┃ ┣ 📂NotFunctionalYet
 ┃ ┃ ┣ 📜Fine_Tuning_distilbert_base_cased_for_Truth_Classification.ipynb
 ┃ ┃ ┣ 📜Fine_Tuning_distilbert_base_uncased_for_Truth_Classification.ipynb
 ┃ ┃ ┣ 📜Fine_Tuning_T5_base_for_Truth_Classification.ipynb
 ┃ ┃ ┣ 📜Fine_Tuning_T5_large_for_Truth_Classification.ipynb
 ┃ ┃ ┗ 📜Fine_Tuning_T5_small_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_Albert_base_v2_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_Albert_large_v2_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_cased_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_large_Cased_for_Truth_Classification (1).ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_large_uncased_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_distilroberta_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_RoBERTa_for_Truth_Classification.ipynb
 ┃ ┗ 📜Fine_Tuning_RoBERTa_large_for_Truth_Classification.ipynb
 ┣ 📂politifact_clean_binarized
 ┃ ┣ 📂NotFunctionalYet
 ┃ ┃ ┣ 📜Fine_Tuning_distilbert_base_cased_for_Truth_Classification.ipynb
 ┃ ┃ ┣ 📜Fine_Tuning_distilbert_base_uncased_for_Truth_Classification.ipynb
 ┃ ┃ ┣ 📜Fine_Tuning_T5_base_for_Truth_Classification.ipynb
 ┃ ┃ ┣ 📜Fine_Tuning_T5_large_for_Truth_Classification.ipynb
 ┃ ┃ ┗ 📜Fine_Tuning_T5_small_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_Albert_base_v2_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_Albert_large_v2_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_cased_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_large_Cased_for_Truth_Classification (1).ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_large_uncased_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_distilroberta_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_RoBERTa_for_Truth_Classification.ipynb
 ┃ ┗ 📜Fine_Tuning_RoBERTa_large_for_Truth_Classification.ipynb
 ┗ 📂politifact_strict_binarized
 ┃ ┣ 📂NotFunctionalYet
 ┃ ┃ ┣ 📜Fine_Tuning_Albert_large_v2_for_Truth_Classification.ipynb
 ┃ ┃ ┣ 📜Fine_Tuning_distilbert_base_cased_for_Truth_Classification.ipynb
 ┃ ┃ ┣ 📜Fine_Tuning_distilbert_base_uncased_for_Truth_Classification.ipynb
 ┃ ┃ ┣ 📜Fine_Tuning_T5_base_for_Truth_Classification.ipynb
 ┃ ┃ ┣ 📜Fine_Tuning_T5_large_for_Truth_Classification.ipynb
 ┃ ┃ ┗ 📜Fine_Tuning_T5_small_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_Albert_base_v2_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_cased_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_large_Cased_for_Truth_Classification (1).ipynb
 ┃ ┣ 📜Fine_Tuning_BERT_large_uncased_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_distilroberta_for_Truth_Classification.ipynb
 ┃ ┣ 📜Fine_Tuning_RoBERTa_for_Truth_Classification.ipynb
 ┃ ┗ 📜Fine_Tuning_RoBERTa_large_for_Truth_Classification.ipynb
```
 The folder above contains a large number of trials created with the goal of obtaining as good of a result as possible. These notebooks were run using Google Colab. Each sub-folder is named after the Dataset it was trying to classify (one of the 3 variants).
```
 📦5. Results and Vizualizations
 ┗ 📜5_a_Results_Vizualization.ipynb
```
 The final folder contains the notebook where all results can be visualised. Sadly, the figures are not displayed in GitHub because of how Plotly functions.

# How to setup the environment

Before delving into constructing the local environment, I have the following mentions:
- All notebooks within the "4. Deep Learning" folder were run within Google Colab. They contain paths to my Google Drive hierarchy, but should be reproducible in Google Colab/locally, but you will have to change paths. Shouldn't require too much effort, but is a hassle nontheless.
- The notebook found at "3. Machine Learning  & AutoML / 3_b_AutoKeras_Truth_Detection_Text_Classification.ipynb" was run using Google Colab.

The notebooks mentioned above should run on Google Colab, but you will either:
- Have to upload the data to the Google Colab env
- Upload the data to your Drive and link it from within the Google Colab env, like I have. I recommend this option.

The 2 ways of environment construction are:
- Create conda env and install requirements using pip
- Pull docker image (TODO)

First, pull the repo.

```
# Using SSH
git clone git@github.com:the-sergiu/TruthDetection.git

# Make sure you have a SSH key-pair setup within Git

```

## Old-fashioned way (conda env)

### Prerequisites
- Anaconda (comes with Anaconda prompt and implicitly pip).
- Possibly an hour to spare.

**1. Open the Conda Prompt**

If the base env is activated, deactivate the base environment. (Should show a (base) tag at the begininng of terminal line.)
```
conda deactivate base
```

**2. Create new Conda environment with the following command, using Python 3.7.**
```
conda create --name py37 python=3.7 
```
Environment will be named *py37* by default. Feel free to change the name.

**3. Navigate to the repo folder within Conda Prompt**

```
cd path/to/repo
```

**4. Install requirements**
Just run the following command:
```
pip install -r requirements.txt
```

Now you play the waiting game.

After installing all the packages (which will take a while), you should be able to run the notebooks found in the 📦1. Data Manipulation,  📦2. Part of Speech Tagging & Named Entity Recognition,  📦5. Results and Vizualizations by just moving them to the "Data" folder. 

Optionally, for cleaner use, you can change the paths to whatever you wish, but make sure that the data csv files can be read from within the notebook.

If you wish to use different Conda comands, click [here](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) for the cheat sheet.


## Docker image

### Prerequisites
- Docker and the implicit setup.

**1. Switch branch to "docker"**

Just run:
```
git checkout docker

```

**3. Just build the docker image**
```
docker build -t td .
```

**4. Run the docker image**
```
docker run -p 8888:8888 td
```
The docker image should run on the 8888 ports because that's the implicit port Jupyter Notebook usually runs on. That's also reflected in the Dockerfile.

