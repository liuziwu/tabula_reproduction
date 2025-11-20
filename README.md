# TabuLa: Harnessing Language Models for Tabular Data Synthesis

[Original Paper](https://arxiv.org/abs/2310.12746)

[Source Code for synprivutil](https://github.com/zhao-zilong/Tabula "Hosted on GitHub")


## Prerequisites 

## Create Virtual Environment

```bash
# Create conda environment
conda create -n tabula python=3.10.19
conda activate tabula
```

## Install Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt
```

The Insurance dataset is also provided within the code. We do not hold the copyright of the dataset; the original dataset can also be downloaded [here](https://www.kaggle.com/datasets/mirichoi0218/insurance). To download the pre-trained model on all datasets used in the paper, download [here](https://drive.google.com/file/d/1_YxelekxY5MXhgn93MYgsZEEfBYAy7h6/view?usp=sharing). Do not forget to create a folder `pretrained-model` and place the downloaded model inside.


### 1. Configure Kaggle Environment (Preparation for Subsequent Dataset Downloads)
- **Step 1: Register a Kaggle account**  
  Visit the link: https://www.kaggle.com/ and complete the account registration.  

- **Step 2: Obtain Kaggle API key**  
  After logging in, go to account settings and download the API key file (named `kaggle.json`).  

- **Step 3: Upload the key to the specified directory**  
  Use Cursor's "File Upload" function to upload `kaggle.json` to the `~/Tabula` directory.  

- **Step 4: Execute configuration commands**  
  Run the following commands to complete Kaggle configuration:  
  ```bash
  # Create Kaggle configuration directory
  mkdir -p ~/.kaggle
  # Move API key to the configuration directory
  mv ~/Tabula/kaggle.json ~/.kaggle/
  # Set permissions (to avoid permission errors)
  chmod 600 ~/.kaggle/kaggle.json
  ```  


### 2. Download Pre-trained Model  
- **Goal**: Download the model to the `./pretrained-model` directory  
- **Commands to execute**:  
  ```bash
  # Create model storage directory
  mkdir -p ./pretrained-model
  # Download the model using gdown (--fuzzy parameter for identifying Google Drive shared links)
  gdown https://drive.google.com/file/d/1_YxelekxY5MXhgn93MYgsZEEfBYAy7h6/view?usp=sharing --fuzzy -O ./pretrained-model/tabula_pretrained_model.pt
  ```  


### 3. Download Various Datasets  

#### 3.1 Insurance Dataset  
- **Goal**: Download and store in the `./insurance` directory  
- **Commands to execute**:  
  ```bash
  # Create dataset directory
  mkdir -p ./insurance
  # Download and unzip the dataset to the specified directory
  kaggle datasets download -d mirichoi0218/insurance --unzip -p ./insurance
  ```  


#### 3.2 Adult Dataset (UCI)  
- **Goal**: Download and store in the `./adult_dataset` directory  
- **Commands to execute**:  
  ```bash
  # Create dataset directory
  mkdir -p ./adult_dataset
  # Download the data file (adult.data)
  wget -P ./adult_dataset https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
  ```  


#### 3.3 Loan Dataset  
- **Goal**: Download and store in the `./loan_dataset` directory  
- **Commands to execute**:  
  ```bash
  # Create dataset directory
  mkdir -p ./loan_dataset
  # Download and unzip the dataset to the specified directory
  kaggle datasets download -d itsmesunil/bank-loan-modelling --unzip -p ./loan_dataset
  ```  


#### 3.4 Covertype Dataset (UCI)  
- **Goal**: Download and unzip to the `./covertype_dataset` directory (final file: `covtype.data`)  
- **Commands to execute**:  
  ```bash
  # Create dataset directory
  mkdir -p ./covertype_dataset
  # Download the compressed dataset
  wget -P ./covertype_dataset https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
  # Unzip the file (generates covtype.data)
  gunzip ./covertype_dataset/covtype.data.gz
  ```  


#### 3.5 Intrusion Dataset (UCI KDD Cup 99)  
- **Goal**: Download and unzip the training and test sets to the `~/Tabula/Real_Datasets/Intrusion` directory  
- **Commands to execute**:  
  ```bash
  # Create dataset directory
  mkdir -p ~/Tabula/Real_Datasets/Intrusion
  # Download training and test sets
  wget -P ~/Tabula/Real_Datasets/Intrusion http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
  wget -P ~/Tabula/Real_Datasets/Intrusion http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled_10_percent.gz
  # Unzip training and test sets
  gunzip ~/Tabula/Real_Datasets/Intrusion/kddcup.data_10_percent.gz
  gunzip ~/Tabula/Real_Datasets/Intrusion/kddcup.testdata.unlabeled_10_percent.gz
  ```  


#### 3.6 King Dataset (House Price Prediction)  
- **Goal**: Download and store in the `~/Tabula/Real_Datasets` directory, and rename to `King_compressed.csv`  
- **Commands to execute**:  
  ```bash
  # Create dataset directory (if it does not exist)
  mkdir -p ~/Tabula/Real_Datasets
  # Download and unzip the dataset
  kaggle datasets download -d harlfoxem/housesalesprediction --unzip -p ~/Tabula/Real_Datasets
  # Rename the dataset (to simplify subsequent loading paths)
  mv ~/Tabula/Real_Datasets/kc_house_data.csv ~/Tabula/Real_Datasets/King_compressed.csv
  ```


### 1. Run the Tabula Model (using the Insurance dataset as an example)
```bash
cd Tabula/run_tabula_code
python Tabula_on_insurance_dataset.py
```


### 2. Run the Tabula_middle_padding Model (using the Adult dataset as an example)
```bash
cd Tabula/run_middle_padding_code
python Tabula_middle_padding_on_adult_dataset.py
```



## Known Issues
When generating synthetic data for the standard version of the Covertype and Intrusion datasets in Python, calling the `sample` method triggers a warning: `The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's attention_mask to obtain reliable results.`, and the progress bar freezes (e.g., `0%|                                               | 0/50000 [02:23<?, ?it/s`).


## bibtex 

```
@inproceedings{zhao2025tabula,
  title={Tabula: Harnessing language models for tabular data synthesis},
  author={Zhao, Zilong and Birke, Robert and Chen, Lydia Y},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={247--259},
  year={2025},
  organization={Springer}
}
```