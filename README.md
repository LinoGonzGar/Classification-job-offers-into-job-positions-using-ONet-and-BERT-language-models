# Classification of job offers into job positions using O*Net and BERT language models

**Abstract:**
Classifying job offers into occupational categories is a fundamental task in candidate-job matching. Comprehensive occupational databases such as O*NET or ESCO provide detailed taxonomies of interrelated job positions, which can be leveraged to match the textual content of job postings. In this work, we explore the effectiveness of fine-tuning existing language models (LMs) to classify job offers with occupational descriptors from O*NET. This enables a more precise assessment of candidate suitability by identifying the specific knowledge and skills required for each position, and facilitates the automation of recruitment processes by helping to mitigate human bias and subjectivity in candidate selection. The best performance was achieved with the DeBERTa model, although the BERT model
also produced strong results. It was also observed that these models tend to reach optimal performance after only a few training epochs, and that training with smaller, balanced datasets has also proven to be effective.

This project trains and compares **three transformer-based models** to classify **job offers** into occupational families defined by the **O*NET database**.

## Models Used
- [BERT](https://huggingface.co/bert-base-uncased) (`bert-base-uncased`)
- [DeBERTa](https://huggingface.co/microsoft/deberta-base) (`microsoft/deberta-base`)
- [RoBERTa](https://huggingface.co/roberta-base) (`roberta-base`)

Each model is fine-tuned on the **job offers dataset** and evaluated with the same metrics and hyperparameter search space.

## Project Structure
```
├── train_bert.py # Train with BERT
├── train_deberta.py # Train with DeBERTa
├── train_roberta.py # Train with RoBERTa
├── dataset_train.csv # Training dataset
├── dataset_test.csv # Test dataset
├── requirements.txt # Dependencies
└── README.md # Documentation
```
## Requirements
Install dependencies with:
```python
pip install -r requirements.txt
```
## Usage

### 1. Prepare datasets

Your CSV files must contain:

- **`description`:** job offer text  
- **`Label`:** target occupational family (integer)

**Example:**
description,Label
"Software developer needed for backend services",5
"Nurse position available at local hospital",2

### 2. Train models

Run the training script for the desired model:
```python
# Train with BERT
python train_bert.py

# Train with DeBERTa
python train_deberta.py

# Train with RoBERTa
python train_roberta.py
```
Each script will:

- Fine-tune the model

- Run Optuna hyperparameter optimization with:

    - **Learning rates:** 2e-5, 3e-5, 5e-5
    
    - **Batch sizes:** 8, 16, 32
    
    - **Weight decay:** 0.01, 0.1
    
    - **Epochs:** 10 (fixed)

- Save predictions to experiments/predictions_<model>.csv

- Log metrics to Weights & Biases

### 3. Weights & Biases

You need a wandb account and to log in locally:
```
wandb login
```
### 4. Outputs

**Metrics:** accuracy, precision, recall, F1 score

**Confusion matrix:** available in wandb dashboard

**Predictions CSV:** file predictions_<model>.csv with:

  - Original job offer text
  
  - True occupational label
  
  - Predicted label


