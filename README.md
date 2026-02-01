🧠 Sentiment Analysis on COVID-19 Tweets using Deep Learning
📌 Project Overview

This project focuses on sentiment analysis of COVID-19 related tweets using Natural Language Processing (NLP) and Deep Learning techniques.
The goal is to classify tweets into different sentiment categories by training and comparing LSTM and GRU models.

📂 Project Structure

Models.ipynb
This Jupyter Notebook includes:

Data loading and exploration

Text preprocessing and cleaning

Tokenization and padding

Building and training LSTM and GRU models

Model evaluation and performance comparison

🗃️ Dataset

Dataset: COVID-19 Tweets

File used:

Corona_NLP_train.csv

Key columns:

OriginalTweet – tweet text

Sentiment – sentiment label (Positive, Negative, Neutral, etc.)

Note: The dataset encoding is latin1.

⚙️ Data Preprocessing

The following preprocessing steps are applied:

Removing URLs and special characters

Converting text to lowercase

Tokenization using Tokenizer

Padding sequences to a fixed length

Splitting data into training and testing sets

🤖 Models Used
🔹 LSTM (Long Short-Term Memory)

Captures long-term dependencies in text sequences

Architecture includes:

Embedding layer

LSTM layer

Dense output layer

Uses:

EarlyStopping

ModelCheckpoint

🔹 GRU (Gated Recurrent Unit)

A lighter and faster alternative to LSTM

Trained and evaluated using the same pipeline for fair comparison

📊 Evaluation & Results

Models are evaluated using:

Accuracy

Loss

Training history visualizations:

Training vs Validation Accuracy

Training vs Validation Loss

A comparison between LSTM and GRU performance is provided.

🛠️ Requirements

To run this project, install the following dependencies:

python >= 3.8
tensorflow
keras
pandas
numpy
matplotlib
scikit-learn

▶️ How to Run

Make sure the dataset is available at:

data/Corona_NLP_train.csv


Open the notebook:

Models.ipynb


Run the cells sequentially

Review the training and evaluation results

🚀 Future Improvements

Experiment with Transformer-based models:

BERT

RoBERTa

Use pretrained word embeddings (GloVe, Word2Vec)

Enhance text preprocessing (lemmatization, stopword removal)

✍️ Author

This project was developed for educational purposes and practical experimentation with NLP and Deep Learning.
