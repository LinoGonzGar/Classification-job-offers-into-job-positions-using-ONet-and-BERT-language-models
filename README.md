# Classification of job offers into job positions using O*Net and BERT language models

**Abstract:**
Classifying job offers into occupational categories is a fundamental task in human resource information systems, as it improves and streamlines indexing, search, and matching between opening and job seekers. Comprehensive
occupational databases such as O*NET or ESCO provide detailed taxonomies of interrelated positions that can be leveraged to align the textual content of postings with occupational categories, thereby facilitating standardization, cross-system interoperability, and access to metadata for each occupation (e.g., tasks, knowledge, skills, and abilities).

In this work, we explore the effectiveness of fine-tuning existing language models (LMs) to classify job offers with occupational descriptors from O*NET. This enables a more precise assessment of candidate suitability by identifying the specific knowledge and skills required for each position, and helps automate recruitment processes by mitigating human bias and subjectivity in candidate selection. We evaluate three representative BERT-like models: BERT, RoBERTa, and DeBERTa. BERT serves as the baseline encoder-only architecture; RoBERTa incorporates advances in pretraining objectives and data scale; and DeBERTa introduces architectural improvements through disentangled attention mechanisms. The best performance was achieved with the DeBERTa model, although the other models also produced strong results, and no statistically significant differences were observed across models. we also find that these models typically reach optimal performance after only a few training epochs, and that training with smaller, balanced datasets is effective. Consequently, comparable results can be obtained with models that require fewer computational resources and less training time, facilitating deployment and practical use.

## Models Used
- [BERT](https://huggingface.co/bert-base-uncased) (`bert-base-uncased`)
- [DeBERTa](https://huggingface.co/microsoft/deberta-base) (`microsoft/deberta-base`)
- [RoBERTa](https://huggingface.co/roberta-base) (`roberta-base`)

Each model is fine-tuned on the **job offers dataset** and evaluated with the same metrics and hyperparameter search space.

## Project Structure
```
├── list_job_scraper.py # List Job URLs from CareerOneStop
├── content_job_scraper.py # Job Details Scraper for CareerOneStop
├── build_dataset.py # Build Unified Dataset from Job Details Scraper
├── balance_dataset.py # Balance Job Descriptions Dataset
├── train_bert.py # Train with BERT
├── train_deberta.py # Train with DeBERTa
├── train_roberta.py # Train with RoBERTa
├── requirements.txt # Dependencies
└── README.md # Documentation
```
## Requirements
Install dependencies with:
```python
pip install -r requirements.txt
```
## Usage

### 1. List Job URLs from CareerOneStop
list_job_scraper.py takes a CSV of occupation categories (https://www.onetonline.org/find/all) and collects job posting URLs for each occupation from CareerOneStop search results, writing one CSV per occupation with a single column: "Job URL".

Usage:
    python list_job_scraper.py \
        --occupations-csv All_Occupations.csv \
        --occupation-column Occupation \
        --pages 5 \
        --radius 100 \
        --location "United States" \
        --max-workers 8 \
        --output-dir ./urls \
        --proxies http://user:pass@host:port,http://host2:3128 \
        --proxies-file proxies.txt \
        --max-requests-per-proxy 10

Example:
    python list_job_scraper.py --pages 12

This is the call for applications to download up to 3,000 job postings (12 pages of 250 postings each) per job category across the US.

### 2. Job Details Scraper for CareerOneStop

content_job_scraper.py takes the CSV files of job posting URLs by occupation from the previous step and extracts the title, company, description, and URL for each job posting. The results are saved in a CSV file for each occupation.

**Usage:**
    python content_job_scraper.py \
        --occupations-csv All_Occupations.csv \
        --occupation-column Occupation \
        --urls-dir ./urls \
        --urls-column "Job URL" \
        --pages-per-occupation all \
        --output-dir ./output \
        --max-workers 10 \
        --timeout 10 \
        --max-retries 5 \
        --proxies http://user:pass@host:port,http://host2:3128 \
        --proxies-file proxies.txt \
        --max-requests-per-proxy 10

**Example:**
    python content_job_scraper.py

This is the call to download the specific data for each job posting downloaded in the previous step.

### 3. Prepare datasets

build_dataset.py merges all per-occupation CSVs produced in Step 2 into a single `dataset.csv`, adds derived fields from the job URL (e.g., `cod_empleo`, `family_empleo`), and encodes a numeric `Label`.The description field is also preprocessed and offers with fewer than the number of words indicated in the call (50 by default) are eliminated.

**Usage:**
    python build_dataset.py \
        --input-dir output \
        --output dataset.csv \
        --min-words 50 \
        --occupations-csv All_Occupations.csv \
        --occupation-column Occupation

**Example:**
    python build_dataset.py

### 4. Balance datasets

balance_dataset.py balances classes via undersampling (down to the minority class size) and performs a stratified train/test split by 'Label'.

**Usage:**
    python balance_dataset.py \
        --input dataset.csv \
        --test-size 0.1 \
        --random-state 42 \
        --train-out dataset_train.csv \
        --test-out dataset_test.csv \
        --report-dir ./reports
**Example:**
    python balance_dataset.py
    
### 5. Train models

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

You need a wandb account and to log in locally:
```
wandb login
```






