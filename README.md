# 🔍 Job Recommender NLP: AI-Powered Job & Talent Matching with Sentence-BERT

## 📘 Overview

This project builds an AI-powered job and resume recommendation engine using Sentence-BERT embeddings to match resumes with job postings and vice versa. It was developed as a demo on Kaggle to showcase how semantic similarity can improve both recruitment and job search over traditional keyword-based systems.

The system processes a subset of the [LinkedIn Job Postings Dataset (2023-24)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) and the [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset), using embeddings to recommend the top 5 matches in both directions (resumes per job, jobs per resume).

### 🔗 Links

- 📘 Kaggle Notebook: [Job Recommender NLP](https://www.kaggle.com/code/muhammadaliasghar01/job-recommender-nlp/)
- 💻 GitHub Repository: [Job-Recommender-NLP](https://github.com/MuhammadAliAsgher/Job-Recommender-NLP/)
- 📄 LinkedIn Job Postings Dataset: [View Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- 📄 Resume Dataset: [View Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

---

## 🔑 Key Features

### 🧠 Bi-Directional Semantic Recommendation Engine

- Matches **1,000 job postings** with **1,000 resumes** using semantic similarity.
- Recommends:
  - **Top 5 resumes** for each job (recruiter perspective).
  - **Top 5 job postings** for each resume (job-seeker perspective).

### ⚙️ Implementation Details

- Built with **Sentence-BERT (all-MiniLM-L6-v2)** for generating embeddings.
- Includes:
  - Text preprocessing with **spaCy** (lemmatization, skill/domain extraction)
  - Embedding generation for job postings and resumes
  - Cosine similarity computation
  - Bidirectional recommendation output
  - Manual validation and average similarity evaluation

---

## 📊 Evaluation

- Processed **1,000 job postings** and **1,000 resumes** from public datasets.
- Evaluated with:
  - Manual inspection of recommended resumes for **Job Postings 0, 1, and 2**
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
- **Job 1’s lower score (0.4035)** indicates dataset limitations (e.g., resumes lacking key terms like "installation").
- The system effectively surfaces relevant matches based on **semantic meaning**, supporting diversity in candidate selection and improving relevance in job recommendations.

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
cd Job-Recommender-NLP
```
### 2. Run the Notebook
- Open `job-recommender-nlp.ipynb` in Jupyter Notebook or JupyterLab.

- Ensure that the datasets (`postings.csv` and `Resume.csv`) are available in the working directory, or update the file paths accordingly.

- Run the cells sequentially to preprocess data, generate embeddings, compute matches, evaluate, and visualize results.

---


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 🙋‍♂️ Contact

Created by **Muhammad Ali Asghar**  
📧 Connect on [LinkedIn](https://www.linkedin.com/in/muhammad-ali-asghar-82b87121b/)  
🌐 Portfolio / GitHub: [github.com/MuhammadAliAsgher](https://github.com/MuhammadAliAsgher)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" />
