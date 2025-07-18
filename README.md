# Detecting Fake News: A Comparative Analysis of SVM and BERT

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Introduction

In the digital age, the rapid spread of fake news poses significant risks to society, from eroding public trust to manipulating public opinion. This project addresses the challenge of misinformation by developing and comparing machine learning models to automatically classify news articles as either "Real" or "Fake" based on their text content.

The project follows a complete data science workflow, from data preprocessing and exploratory analysis to model training, evaluation, and a final qualitative analysis on new, unseen data.

***
## 2. Project Goal

The primary objective is to develop a high-accuracy classification system by implementing and evaluating two distinct approaches:
* **Classic Machine Learning**: Utilizing a Support Vector Machine (SVM) model with TF-IDF features to establish a strong baseline.
* **Deep Learning**: Applying the state-of-the-art BERT (`bert-base-uncased`) model to leverage deep contextual understanding of the text.

***
## 3. Dataset

The dataset is a composite of articles from two main categories:
* **True News**: Sourced from reputable media outlets like Reuters and The New York Times.
* **Fake News**: Sourced from known misinformation/extremist websites and public datasets.

After a thorough cleaning process—which involved removing 29 missing values and 10,012 duplicate records—the final dataset consists of **68,380 articles**. The dataset is well-balanced, with 50.5% real news and 49.5% fake news.

***
## 4. Methodology & Workflow

The project adhered to a structured machine learning workflow:

1.  **Data Preprocessing**:
    * Assigned binary labels: `1` for Real News, `0` for Fake News.
    * Removed duplicates and null values.
    * Engineered new features like `word_count` and `text_length` to aid analysis.
    * Filtered out noisy articles with fewer than 6 words.

2.  **Exploratory Data Analysis (EDA)**:
    * Analysis revealed that real news articles tend to be longer (average 537 words) than fake news articles (average 429 words), confirming that text length is a useful, albeit insufficient, feature.

3.  **Model Building & Training**:
    * **SVM**:
        * Features were generated using TF-IDF (`max_features=10000`, `ngrams=(1,2)`) combined with normalized numerical features.
        * `GridSearchCV` was used to find the optimal hyperparameters: `kernel='rbf'`, `C=10.0`, and `gamma='scale'`.
    * **BERT**:
        * The `bert-base-uncased` model was implemented using the `transformers` and `torch` libraries.
        * The model was trained for 3 epochs with a batch size of 16 and a learning rate of 3e-5.

4.  **Evaluation**: Models were evaluated on a 20% test split using standard classification metrics, followed by a qualitative test on new data.

***
## 5. Results & Analysis

Both models performed exceptionally well on the test set, but BERT demonstrated superior generalization capabilities in real-world testing.

### Performance on Test Set

| Model  | Accuracy | F1-Score (Fake) | F1-Score (Real) | Overall F1-Score |
| :----- | :------: | :-------------: | :-------------: | :----------------: |
| **SVM** | 0.95 | 0.95 | 0.95 | 0.95 |
| **BERT** | **0.98** | **0.98** | **0.98** | **0.98** |

### Qualitative Test on 10 New Articles

To assess real-world performance, the models were tested on 10 new articles (5 real, 5 fake).

* **SVM**: Achieved **60% accuracy**, crucially misclassifying 4 out of 5 real news articles as fake. This highlights its limited generalization ability when faced with unseen data patterns.
* **BERT**: Achieved **90% accuracy**, only making one error on a real news article about NASA. This showcases its robust contextual understanding and superior generalization.

***
## 6. Tech Stack

* **Programming Language**: Python
* **Libraries**:
    * Data Manipulation: Pandas, NumPy
    * ML / DL: Scikit-learn, PyTorch, Transformers
    * Visualization: Matplotlib, Seaborn
    * Notebook: Jupyter Notebook

***
## 7. Conclusion & Future Work

While both models were highly effective, BERT's ability to understand deep semantic context makes it the superior choice for complex NLP tasks like fake news detection. The model's primary limitation is its high computational cost.

Future development directions include:
* Experimenting with larger transformer models like RoBERTa or DeBERTA.
* Implementing model interpretability techniques like LIME/SHAP to understand BERT's decisions.
* Deploying the fine-tuned BERT model as a real-time API or web application for practical use.

***
## 8. Author(s)

* **Đỗ Trần Sáng** ([@Meanless])
