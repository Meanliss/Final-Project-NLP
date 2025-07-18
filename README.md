# Detecting Fake News: A Comparative Analysis of SVM and BERT

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Introduction

[cite_start]In the digital age, the rapid spread of fake news poses significant risks to society, from eroding public trust to manipulating public opinion[cite: 10]. [cite_start]This project addresses the challenge of misinformation by developing and comparing machine learning models to automatically classify news articles as either "Real" or "Fake" based on their text content[cite: 11].

[cite_start]The project follows a complete data science workflow, from data preprocessing and exploratory analysis to model training, evaluation, and a final qualitative analysis on new, unseen data[cite: 12].

## 2. Project Goal

[cite_start]The primary objective is to develop a high-accuracy classification system by implementing and evaluating two distinct approaches[cite: 14]:
* [cite_start]**Classic Machine Learning**: Utilizing a Support Vector Machine (SVM) model with TF-IDF features to establish a strong baseline[cite: 15].
* [cite_start]**Deep Learning**: Applying the state-of-the-art BERT (`bert-base-uncased`) model to leverage deep contextual understanding of the text[cite: 16].

## 3. Dataset

The dataset is a composite of articles from two main categories:
* [cite_start]**True News**: Sourced from reputable media outlets like Reuters and The New York Times[cite: 18].
* [cite_start]**Fake News**: Sourced from known misinformation/extremist websites and public datasets[cite: 18].

[cite_start]After a thorough cleaning process—which involved removing 29 missing values and 10,012 duplicate records—the final dataset consists of **68,380 articles**[cite: 19, 24]. [cite_start]The dataset is well-balanced, with 50.5% real news and 49.5% fake news[cite: 19].

## 4. Methodology & Workflow

The project adhered to a structured machine learning workflow:

1.  **Data Preprocessing**:
    * [cite_start]Assigned binary labels: `1` for Real News, `0` for Fake News[cite: 23].
    * [cite_start]Removed duplicates and null values[cite: 24].
    * [cite_start]Engineered new features like `word_count` and `text_length` to aid analysis[cite: 26].
    * [cite_start]Filtered out noisy articles with fewer than 6 words[cite: 27].

2.  **Exploratory Data Analysis (EDA)**:
    * [cite_start]Analysis revealed that real news articles tend to be longer (average 537 words) than fake news articles (average 429 words), confirming that text length is a useful, albeit insufficient, feature[cite: 32].

3.  **Model Building & Training**:
    * **SVM**:
        * [cite_start]Features were generated using TF-IDF (`max_features=10000`, `ngrams=(1,2)`) combined with normalized numerical features[cite: 37].
        * [cite_start]`GridSearchCV` was used to find the optimal hyperparameters: `kernel='rbf'`, `C=10.0`, and `gamma='scale'`[cite: 38].
    * **BERT**:
        * [cite_start]The `bert-base-uncased` model was implemented using the `transformers` and `torch` libraries[cite: 16, 44].
        * [cite_start]The model was trained for 3 epochs with a batch size of 16 and a learning rate of 3e-5[cite: 45].

4.  **Evaluation**: Models were evaluated on a 20% test split using standard classification metrics, followed by a qualitative test on new data.

## 5. Results & Analysis

Both models performed exceptionally well on the test set, but BERT demonstrated superior generalization capabilities in real-world testing.

### Performance on Test Set

| Model | Accuracy | F1-Score (Fake) | F1-Score (Real) | Overall F1-Score |
| :---- | :------: | :-------------: | :-------------: | :----------------: |
| **SVM** | [cite_start]0.95 [cite: 40] | [cite_start]0.95 [cite: 40] | [cite_start]0.95 [cite: 40] | [cite_start]0.95 [cite: 58] |
| **BERT** | [cite_start]**0.98** [cite: 50] | [cite_start]**0.98** [cite: 50] | [cite_start]**0.98** [cite: 50] | [cite_start]**0.98** [cite: 58] |

### Qualitative Test on 10 New Articles

[cite_start]To assess real-world performance, the models were tested on 10 new articles (5 real, 5 fake)[cite: 53].

* **SVM**: Achieved **60% accuracy**, crucially misclassifying 4 out of 5 real news articles as fake. [cite_start]This highlights its limited generalization ability when faced with unseen data patterns[cite: 54].
* **BERT**: Achieved **90% accuracy**, only making one error on a real news article about NASA. [cite_start]This showcases its robust contextual understanding and superior generalization[cite: 55].

## 6. Tech Stack

* **Programming Language**: Python
* **Libraries**:
    * Data Manipulation: Pandas, NumPy
    * ML / DL: Scikit-learn, PyTorch, Transformers
    * Visualization: Matplotlib, Seaborn
    * Notebook: Jupyter Notebook

## 7. Setup & Installation

To run this project locally, follow these steps:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd your-repo-name
    ```
3.  Install the required dependencies. It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

***Note:*** *To generate the `requirements.txt` file, you can run `pip freeze > requirements.txt` in your terminal.*

## 8. Usage

All the analysis, modeling, and evaluation steps are detailed in the `Fake_News_Detection.ipynb` Jupyter Notebook. Open the notebook and run the cells sequentially to reproduce the results.

## 9. Conclusion & Future Work

[cite_start]While both models were highly effective, BERT's ability to understand deep semantic context makes it the superior choice for complex NLP tasks like fake news detection[cite: 59]. [cite_start]The model's primary limitation is its high computational cost[cite: 62].

Future development directions include:
* [cite_start]Experimenting with larger transformer models like RoBERTa or DeBERTA[cite: 65].
* [cite_start]Implementing model interpretability techniques like LIME/SHAP to understand BERT's decisions[cite: 66].
* [cite_start]Deploying the fine-tuned BERT model as a real-time API or web application for practical use[cite: 68].

## 10. Author(s)

* [cite_start]**Đỗ Trần Sáng** ([@your-github-username](https://github.com/your-github-username)) [cite: 2]
