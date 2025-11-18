Tabula Reproduction README
Project Overview
This project is based on the official Tabula code repository, reproducing the core functionality of the Tabula table data generation model. It supports comparative testing of both the standard version and the middle padding version. The model is compatible with multiple typical tabular datasets (covering both classification and regression tasks) and provides a complete workflow including data processing, model training, and result evaluation.
Key Information
Paper Reference: Tabula: Table Data Generation with Pre-trained Language Models
Code Repository: https://github.com/zhao-zilong/Tabula
Supported Versions:
Standard Version: Import via from tabula import Tabula
Middle Padding Version: Import via from tabula_middle_padding import Tabula (for testing the middle padding strategy)
Supported Tasks: Classification tasks (e.g., income classification, vegetation classification, intrusion detection) and regression tasks (e.g., insurance cost prediction, housing price prediction)
Hardware and Software Requirements
Hardware Requirements
GPU: Memory ≥ 16GB (RTX 3090/4090 recommended, supporting multi-GPU parallelism)
CPU: ≥ 8 cores (to accelerate data preprocessing and evaluation)
RAM: ≥ 32GB (to adapt to large-scale datasets)
Software Requirements
Operating System: Linux or Linux-like systems
Python Version: ≥ 3.9
Core Dependencies: datasets, numpy, pandas, scikit-learn, torch, tqdm, transformers, gdown (see the official repository for version requirements)
Dataset Preparation
Default Dataset Path: Tabula/Real_Datasets/ (compressed datasets are included in the code repository)
Dataset Types: Covering classification and regression tasks, including tabular data of varying scales and feature dimensions
Dataset Validation: Navigate to the Tabula directory and execute ls Real_Datasets/ to confirm the existence of the target dataset files
Reproduction Steps
General Workflow (Applicable to All Datasets)
Create an execution script in the Tabula directory, implementing the logic in the order of "Environment Configuration → Data Loading → Model Training → Synthetic Data Generation → Evaluation".
Run the script to complete reproduction: python script_name.py
1. Standard Version Reproduction
Environment Configuration: Specify available GPUs (adjust the ID based on hardware).
Data Loading and Preprocessing: Load the target dataset, define core categorical columns, and convert categorical columns to object type.
Model Initialization: Initialize the model using the tabula.Tabula class, configure parameters (e.g., base model, batch size, training epochs), set an appropriate tokenizer length, and load pre-trained model weights.
Model Training: Call the fit method to train the model, and save the trained model weights to a specified directory after training.
Synthetic Data Generation: Call the sample method to generate synthetic data of the same size as the original data, and save it to the dataset directory.
Result Evaluation: Preprocess real data and synthetic data (e.g., categorical column numerical mapping, missing value imputation), use task-specific evaluation models (classifiers for classification tasks, regressors for regression tasks) to evaluate the effectiveness of synthetic data, and select core metrics based on task type.
2. Middle Padding Version Reproduction
The core differences lie in module import and pre-trained model loading; other steps are consistent with the standard version:
Module Import: Replace the standard version module with tabula_middle_padding.Tabula.
Model Initialization: Skip pre-trained model loading (the middle padding strategy does not require pre-trained weights), and keep other parameters (batch size, training epochs, etc.) consistent with the standard version.
The training, synthetic data generation, and evaluation processes are identical to the standard version. The save paths for synthetic data and model weights need to be differentiated.
3. Dataset Adaptation Notes
Classification Tasks: Define categorical feature columns and label columns; the main evaluation metric is weighted F1-score.
Regression Tasks: Define discrete feature columns and target value columns; the main evaluation metrics are R² coefficient of determination and RMSE.
Large-Scale Datasets: Appropriately adjust the batch size (e.g., reduce to 16) to adapt to hardware memory.
High-Dimensional Feature Datasets: Correspondingly adjust the tokenizer length to avoid feature encoding truncation.
Result Evaluation Criteria
Task Type	Core Metric	Qualification Standard
Regression	R² Coefficient of Determination	≥ 0.7 (≥ 0.8 for some datasets)
Classification	Weighted F1-score	≥ 0.75 (≥ 0.7 for some datasets)
Known Issues
When generating synthetic data for the standard version of the Covertype and Intrusion datasets, calling the sample method triggers a warning: The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's attention_mask to obtain reliable results., and the progress bar freezes (e.g., 0%| | 0/50000 [02:23<?, ?it/s).