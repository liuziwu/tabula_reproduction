# Tabula Paper Reproduction README

## Project Overview

This project is based on the official Tabula code repository, reproducing the core functionality of the Tabula table data generation model. It supports comparative testing of both the standard version and the middle padding version. The model is compatible with multiple typical tabular datasets (covering both classification and regression tasks) and provides a complete workflow including data processing, model training, and result evaluation.

## Key Information

- **Paper Reference**: *Tabula: Table Data Generation with Pre-trained Language Models*
- **Code Repository**: https://github.com/zhao-zilong/Tabula
- **Supported Versions**:
  - **Standard Version**: Import via `from tabula import Tabula`
  - **Middle Padding Version**: Import via `from tabula_middle_padding import Tabula` (for testing the middle padding strategy)
- **Supported Tasks**: Classification tasks (e.g., income classification, vegetation classification, intrusion detection) and regression tasks (e.g., insurance cost prediction, housing price prediction)

## Hardware and Software Requirements

### Hardware Requirements
- **GPU**: Memory ≥ 16GB (RTX 3090/4090 recommended, supporting multi-GPU parallelism)
- **CPU**: ≥ 8 cores (to accelerate data preprocessing and evaluation)
- **RAM**: ≥ 32GB (to adapt to large-scale datasets)

### Software Requirements
- **Operating System**: Linux or Linux-like systems
- **Python Version**: ≥ 3.9
- **Core Dependencies**: `datasets`, `numpy`, `pandas`, `scikit-learn`, `torch`, `tqdm`, `transformers`, `gdown` (see the official repository for version requirements)

## Dataset Preparation

- **Default Dataset Path**: `Tabula/Real_Datasets/` (compressed datasets are included in the code repository)
- **Dataset Types**: Covering classification and regression tasks, including tabular data of varying scales and feature dimensions
- **Dataset Validation**: Navigate to the `Tabula` directory and execute `ls Real_Datasets/` to confirm the existence of the target dataset files

## Reproduction Steps

### General Workflow (Applicable to All Datasets)

1. Create an execution script in the `Tabula` directory, implementing the logic in the order of:
   - Environment Configuration → Data Loading → Model Training → Synthetic Data Generation → Evaluation
2. Run the script to complete reproduction:
   ```bash
   python script_name.py