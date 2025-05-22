# 🔍 Job Recommender NLP: AI-Powered Job Matching with Sentence-BERT

## 📘 Overview

This project builds an AI-powered job recommendation engine using Sentence-BERT embeddings to match job seekers’ resumes with job postings. It was developed as a demo on Kaggle to showcase the application of NLP in improving job matching beyond traditional keyword-based methods.

The system processes a subset of the [LinkedIn Job Postings Dataset (2023-24)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) and the [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset), using embeddings to recommend the top 5 resumes for each job posting based on semantic similarity. It includes data preprocessing, embedding generation, similarity matching, evaluation, and visualization.

### 🔗 Links

- 📘 Kaggle Notebook: [Job Recommender NLP](https://www.kaggle.com/code/muhammadaliasghar01/job-recommender-nlp/)
- 💻 GitHub Repository: [Job-Recommender-NLP](https://github.com/MuhammadAliAsgher/Job-Recommender-NLP)
- 📄 LinkedIn Job Postings Dataset: [View Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- 📄 Resume Dataset: [View Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

---

## 🔑 Key Features

### 🧠 Job Recommendation Engine

- Matches **200 job postings** with **200 resumes** using semantic similarity.
- Recommends the **top 5 resumes** per job posting based on cosine similarity scores.

### ⚙️ Implementation Details

- Built with **Sentence-BERT (all-MiniLM-L6-v2)** for generating embeddings.
- Includes:
  - Text preprocessing with **spaCy** (lemmatization, skill/domain extraction)
  - Embedding generation for job postings and resumes
  - Cosine similarity computation for matching
  - Manual validation and average similarity score evaluation

---

## 📊 Evaluation

- Processed **200 job postings** and **200 resumes** from public datasets.
- Evaluated with:
  - Manual inspection of matches for **Jobs 0–2**
  - Average similarity scores:

| Metric                 | Score  |
|------------------------|--------|
| Overall Avg Similarity | 0.5853 |
| Job 0 Avg Similarity   | 0.5882 |
| Job 1 Avg Similarity   | 0.4035 |
| Job 2 Avg Similarity   | 0.5161 |


---

## 📋 Results

- **Job 0 and Job 2** show strong matches with high similarity scores.
- **Job 1’s lower score (0.4035)** indicates dataset limitations (e.g., lack of relevant skills like installation in resumes).
- The system effectively surfaces relevant matches beyond keyword-based methods, promoting diversity in candidate selection.

---

## 📁 Repository Contents

- `job-recommender-nlp.ipynb`: Jupyter Notebook with the full implementation
- `LICENSE`: MIT License for open-source usage
- 📎 Kaggle Notebook: [View on Kaggle](https://www.kaggle.com/code/muhammadaliasghar01/job-recommender-nlp/)

---

## 🚀 How to Run

### ✅ Prerequisites

- Python **3.10+**
- Required libraries:
  - `numpy`
  - `pandas`
  - `spacy`
  - `sentence-transformers`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Install dependencies:

```bash
pip install numpy pandas spacy sentence-transformers scikit-learn matplotlib seaborn
```

Install the spaCy model:

```bash
python -m spacy download en_core_web_sm
```
---
## 🧾 Steps
### 1. Clone the Repository
```bash
git clone https://github.com/MuhammadAliAsgher/Job-Recommender-NLP
```
```bash
cd Digit-Recognition-Neural-Network
```
### 2. Run the Notebook
- Open `job-recommender-nlp.ipynb` in Jupyter Notebook or JupyterLab.

- Ensure that the datasets (`postings.csv` and `Resume.csv`) are available in the working directory, or update the file paths accordingly.

- Run the cells sequentially to preprocess data, generate embeddings, compute matches, evaluate, and visualize results.

---
