# Fake Review Detection Using Machine Learning

## Project Overview

The Fake Review Detection project uses Natural Language Processing (NLP) and machine learning to analyze textual reviews and classify them into two categories: Fake and Genuine.

The project uses the IMDB movie review dataset, TF-IDF for text feature extraction, and Logistic Regression for classification. It also evaluates model performance using classification metrics and visualizations.

The implementation is available as a Google Colab notebook.

## Objectives

- Process and analyze textual reviews.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classification model.
- Predict the category of a user-provided review.
- Evaluate performance using accuracy, precision, recall, and F1-score.
- Visualize model performance using a confusion matrix, ROC curve, and learning curves.

## Technologies Used

- **Python** — Core programming language
- **Pandas** — Dataset loading and manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — Feature extraction, model training, and evaluation
- **TF-IDF Vectorizer** — Text feature extraction
- **Logistic Regression** — Classification algorithm
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualizations
- **Google Colab** — Development environment

## Dataset

The project uses the **IMDB Dataset.csv** file, containing movie reviews and their sentiment labels.

The notebook samples 10,000 reviews for computational efficiency.

Expected dataset structure:

| Column | Description |
|---|---|
| `review` | Text of the movie review |
| `sentiment` | Sentiment label, such as positive or negative |

The CSV file must be uploaded to Google Colab and made available at `/content/IMDB Dataset.csv`.

**Important limitation:** The current notebook maps positive sentiment to “Genuine” and negative sentiment to “Fake.” Sentiment polarity does not establish whether a review is authentic or fabricated. Therefore, this implementation demonstrates sentiment-based text classification rather than validated fake-review detection. Reliable fake-review detection requires a dataset with actual fake/genuine labels.

## Methodology

### 1. Data Loading

The IMDB dataset is loaded into a Pandas DataFrame. A sample of 10,000 reviews is selected to reduce computational requirements.

### 2. Label Encoding

The sentiment labels are converted into numerical values:

- `1` — Positive sentiment, labeled as Genuine in the current notebook
- `0` — Negative sentiment, labeled as Fake in the current notebook

These labels reflect the notebook's implementation and should not be interpreted as verified authenticity labels.

### 3. Text Preprocessing and Feature Extraction

The `TfidfVectorizer` converts textual reviews into numerical feature vectors.

The notebook uses:

- English stop-word removal
- A maximum vocabulary size of 300 features
- TF-IDF weighting to represent the importance of words in reviews

These numerical features are used as input to the machine learning model.

### 4. Dataset Splitting

The dataset is split into:

- **80% training data** — Used to train the model
- **20% testing data** — Used to evaluate its performance

The split uses `random_state=42` for reproducibility.

### 5. Model Training

Logistic Regression is trained on the TF-IDF feature matrix.

The algorithm learns relationships between textual features and the supplied labels, then predicts the label of an unseen review.

### 6. Review Prediction

A custom `predict_review()` function accepts user-provided text, transforms it using the fitted TF-IDF vectorizer, and passes the resulting features to the trained classifier.

The model returns the predicted category according to the current label mapping.

## Machine Learning Algorithm

### Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a sample belonging to a class.

In this project:

1. TF-IDF converts reviews into numerical vectors.
2. Logistic Regression learns patterns associated with the training labels.
3. The trained model predicts labels for test reviews.
4. The same pipeline classifies newly entered reviews.

The model uses a maximum of 1,000 iterations and the `liblinear` solver.

## Model Evaluation

The notebook generates the following evaluation outputs:

| Metric or Visualization | Purpose |
|---|---|
| Accuracy | Measures the proportion of correct predictions |
| Classification report | Displays precision, recall, and F1-score |
| Confusion matrix | Shows correct and incorrect classifications |
| Training vs. test accuracy | Compares performance across training-set sizes |
| ROC curve | Evaluates classification behavior across thresholds |
| AUC | Summarizes the ROC curve's discrimination performance |
| Class distribution plot | Visualizes the distribution of the two mapped labels |

The notebook also exports the TF-IDF feature matrix and test predictions as CSV files.

Actual metric values should be taken from the results generated when the notebook is executed.

## How to Run the Project

1. Download `Fake_Review_Detection.ipynb`.
2. Open [Google Colab](https://colab.research.google.com/).
3. Select **File → Upload notebook**.
4. Upload the notebook.
5. Run the cells sequentially.
6. Upload `IMDB Dataset.csv` when prompted.
7. Review the classification report and generated visualizations.
8. Enter a review when prompted to test the prediction function.

## Output Files

The notebook generates:

- `tfidf_results.csv` — TF-IDF feature matrix
- `logreg_predictions.csv` — Actual and predicted test labels

The notebook also displays the confusion matrix, ROC curve, learning curves, and class distribution plot.

## Limitations

- The current labels represent sentiment, not verified review authenticity.
- TF-IDF is limited to a vocabulary of 300 features.
- The model may not generalize to other review platforms or product categories.
- The current implementation uses a single classification algorithm.
- The prediction function's Fake/Genuine output is not reliable evidence of authenticity without appropriately labeled training data.

## Future Scope

- Use datasets containing verified fake and genuine reviews.
- Include reviewer behavior and metadata as additional features.
- Compare Logistic Regression with SVM, Naive Bayes, and other classifiers.
- Experiment with more advanced NLP models such as BERT or RoBERTa.
- Improve preprocessing and evaluate generalization on unseen datasets.
- Develop a web interface for interactive review analysis.

## Project Information

**Project Title:** Fake Review Detection Using Machine Learning

**Domain:** Machine Learning and Natural Language Processing

**Dataset:** IMDB Movie Reviews

**Model:** Logistic Regression

**Feature Extraction:** TF-IDF

**Platform:** Google Colab

---

Developed as an academic project to explore text classification, natural language processing, and machine learning evaluation.
