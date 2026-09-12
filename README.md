# Arabic Sentiment Analysis

## Overview

This project focuses on sentiment classification of Arabic hotel reviews.

The goal is to classify each review as either **positive** or **negative** using different Machine Learning and Deep Learning approaches.

## Dataset

The dataset contains Arabic hotel reviews with the following main information:

- Review ID
- Hotel Name
- Sentiment
- User Type
- Room Type
- Number of Nights
- Review Text

The training dataset contains **84,443 reviews**.

The sentiment classes are almost perfectly balanced:

- Negative: ~50.05%
- Positive: ~49.95%

## Data Exploration & Preprocessing

Some basic exploration was performed on the reviews.

- Average review length: ~24 words
- Median review length: 17 words
- Review length range: 2–614 words
- 83,229 unique reviews

For text cleaning, the following preprocessing steps were applied:

- Removing non-Arabic characters
- Removing extra whitespace
- Stripping leading and trailing spaces

## Machine Learning Approaches

For traditional Machine Learning models, the text was converted into numerical features using **TF-IDF**.

Two TF-IDF configurations were explored:

- 10,000 features
- 20,000 features with **unigrams and bigrams**

The following models were evaluated:

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes

## Deep Learning Approaches

For Deep Learning models, the reviews were converted into sequences of tokens.

The main settings were:

- Vocabulary size: 10,000
- Sequence length: 100 tokens
- Padding: Post-padding
- Embedding dimension: 64

The following architectures were evaluated:

- Simple RNN
- LSTM
- GRU
- Transformer-based architecture using Multi-Head Attention

## Model Performance

| Model | Validation Accuracy |
|---|---:|
| Logistic Regression + TF-IDF (Bigrams) | 93.77% |
| Linear SVM | 93.00% |
| Multinomial Naive Bayes | 90.84% |
| Simple RNN | 78.86% |
| LSTM | **94.22%** |
| GRU | 93.75% |
| Transformer-based Model | 92.01% |

## Key Results

- Evaluated 7 Machine Learning and Deep Learning models.
- The best validation accuracy was achieved by the LSTM model: **94.22%**.
- Logistic Regression with TF-IDF unigrams and bigrams achieved **93.77%**.
- The dataset contains approximately 84K Arabic hotel reviews with balanced sentiment classes.

### Best Model

The **LSTM** achieved the highest validation accuracy of **94.22%** among the evaluated models.

## Error Analysis

Misclassified validation reviews were inspected to better understand cases where the LSTM model made incorrect predictions.

Examples included reviews where the actual sentiment differed from the predicted sentiment.

## Final Prediction

The LSTM model was used to generate predictions for the test dataset.

The test set contains **21,222 reviews**.

The final submission file contains:

- `Review_ID`
- `Sentiment`

The predictions were saved in:

`submission.csv`

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

## Project Pipeline

**Text Cleaning → Tokenization → Padding → LSTM → Sentiment Prediction**
