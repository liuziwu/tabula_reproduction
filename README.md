# TabuLa: Harnessing Language Models for Tabular Data Synthesis

[Original Paper](https://arxiv.org/abs/2310.12746)  
[Source Code ](https://github.com/zhao-zilong/Tabula "Hosted on GitHub")  


## I. Environment Preparation

### 1.1 Creating a Virtual Environment
```bash
# Create a conda environment
conda create -n tabula python=3.10.19
conda activate tabula
```

### 1.2 Installing Dependencies
```bash
# Install basic dependencies
pip install -r requirements.txt
```


## II. Data and Model Preparation

### 2.1 Configuring the Kaggle Environment (for subsequent dataset downloads)
- **Step 1: Register for a Kaggle account**  
  Visit the link: https://www.kaggle.com/ and complete the account registration.  

- **Step 2: Obtain a Kaggle API key**  
  After logging in, go to account settings and download the API key file (named `kaggle.json`).  

- **Step 3: Upload the key to the specified directory**  
  Use Cursor's "File Upload" function to upload `kaggle.json` to the `~/Tabula` directory.  

- **Step 4: Execute configuration commands**  
  Run the following commands to complete Kaggle configuration:  
  ```bash
  # Create a Kaggle configuration directory
  mkdir -p ~/.kaggle
  # Move the API key to the configuration directory
  mv ~/Tabula/kaggle.json ~/.kaggle/
  # Set permissions (to avoid permission errors)
  chmod 600 ~/.kaggle/kaggle.json
  ```  


### 2.2 Downloading the Pretrained Model
- **Objective**: Download the model to the `./pretrained-model` directory  
- **Command to execute**:  
  ```bash
  # Create a directory for storing the model
  mkdir -p ./pretrained-model
  # Download the model using gdown (the --fuzzy parameter is used to identify Google Drive share links)
  gdown https://drive.google.com/file/d/1_YxelekxY5MXhgn93MYgsZEEfBYAy7h6/view?usp=sharing --fuzzy -O ./pretrained-model/tabula_pretrained_model.pt
  ```  


### 2.3 Downloading Various Datasets

#### 2.3.1 Insurance Dataset
- **Objective**: Download and store in the `./insurance` directory  
- **Command to execute**:  
  ```bash
  # Create a dataset directory
  mkdir -p ./insurance
  # Download and unzip the dataset to the specified directory
  kaggle datasets download -d mirichoi0218/insurance --unzip -p ./insurance
  ```  


#### 2.3.2 Adult Dataset
- **Objective**: Download and store in the `./adult_dataset` directory  
- **Command to execute**:  
  ```bash
  # Create a dataset directory
  mkdir -p ./adult_dataset
  # Download the data file (adult.data)
  wget -P ./adult_dataset https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
  ```  


#### 2.3.3 Loan Dataset
- **Objective**: Download and store in the `./loan_dataset` directory  
- **Command to execute**:  
  ```bash
  # Create a dataset directory
  mkdir -p ./loan_dataset
  # Download and unzip the dataset to the specified directory
  kaggle datasets download -d itsmesunil/bank-loan-modelling --unzip -p ./loan_dataset
  ```  


#### 2.3.4 Covertype Dataset
- **Objective**: Download and unzip to the `./covertype_dataset` directory  
- **Command to execute**:  
  ```bash
  # Create a dataset directory
  mkdir -p ./covertype_dataset
  # Download the compressed dataset
  wget -P ./covertype_dataset https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
  # Unzip the file (generates covtype.data)
  gunzip ./covertype_dataset/covtype.data.gz
  ```  


#### 2.3.5 Intrusion Dataset
- **Objective**: Download and unzip the training and test sets to the `~/Tabula/Real_Datasets/Intrusion` directory  
- **Command to execute**:  
  ```bash
  # Create a dataset directory
  mkdir -p ~/Tabula/Real_Datasets/Intrusion
  # Download the training and test sets
  wget -P ~/Tabula/Real_Datasets/Intrusion http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
  wget -P ~/Tabula/Real_Datasets/Intrusion http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled_10_percent.gz
  # Unzip the training and test sets
  gunzip ~/Tabula/Real_Datasets/Intrusion/kddcup.data_10_percent.gz
  gunzip ~/Tabula/Real_Datasets/Intrusion/kddcup.testdata.unlabeled_10_percent.gz
  ```  


#### 2.3.6 King Dataset 
- **Objective**: Download and store in the `~/Tabula/Real_Datasets` directory, and name it `King_compressed.csv`  
- **Command to execute**:  
  ```bash
  # Create a dataset directory (if it does not exist)
  mkdir -p ~/Tabula/Real_Datasets
  # Download and unzip the dataset
  kaggle datasets download -d harlfoxem/housesalesprediction --unzip -p ~/Tabula/Real_Datasets
  # Rename the dataset (to simplify subsequent loading paths)
  mv ~/Tabula/Real_Datasets/kc_house_data.csv ~/Tabula/Real_Datasets/King_compressed.csv
  ```  


## III. Model Running Examples

### 3.1 Running the Tabula Model (using the Insurance Dataset as an example)
```bash
cd Tabula/run_tabula_code
python Tabula_on_insurance_dataset.py
```


### 3.2 Running the Tabula_middle_padding Model (using the Adult Dataset as an example)
```bash
cd Tabula/run_middle_padding_code
python Tabula_middle_padding_on_adult_dataset.py
```


## IV. Known Issues
When generating synthetic data for the standard versions of the Covertype and Intrusion datasets in Python, calling the `sample` method triggers a warning: `The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's attention_mask to obtain reliable results.`, and the progress bar freezes (e.g.: `0%|                                               | 0/50000 [02:23<?, ?it/s`).


## V. References
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