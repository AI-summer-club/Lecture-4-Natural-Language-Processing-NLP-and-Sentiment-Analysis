# Lecture 4: Natural Language Processing (NLP) and Sentiment Analysis

## Introduction
Natural Language Processing (NLP) is a field of artificial intelligence that deals with understanding, processing, and generating human language. In this lecture and lab, we will explore the basics of NLP, including text preprocessing, tokenization, word embeddings, and sentiment analysis using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks.

## Notes

- **ChatGPT Encouraged:** Feel free to use ChatGPT during this lab session to ask questions about installation procedures or Python code.
- **Submit source code on D2L for Credit:**

## Lecture 4: Natural Language Processing (NLP) Basics

### Text Preprocessing
Text preprocessing is an essential step in NLP tasks. It involves cleaning and preparing the text data for further analysis. Common preprocessing steps include:
- Removing HTML tags, URLs, and special characters
- Converting text to lowercase or uppercase
- Removing stop words (e.g., "the", "and", "is")
- Stemming or lemmatization (reducing words to their root form)

### Tokenization
Tokenization is the process of splitting text into smaller units called tokens. These tokens can be words, phrases, or even individual characters, depending on the task.

### Word Embeddings
Word embeddings are dense vector representations of words that capture semantic and syntactic relationships between them. Popular word embedding techniques include Word2Vec and GloVe.

### Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) Networks
RNNs are a type of neural network designed to handle sequential data, such as text. However, they suffer from the vanishing gradient problem when dealing with long sequences. LSTMs are a special kind of RNN that can learn long-term dependencies more effectively, making them well-suited for NLP tasks.

### Applications of NLP
NLP has numerous applications, including sentiment analysis, language translation, text summarization, and question answering.

## Lab 4: Building a Simple Sentiment Analysis Model for Movie Reviews
In this lab, we will build a sentiment analysis model to classify movie reviews as positive or negative using an LSTM network. Please write your code using the following libraries:

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```

### Dataset
We will use the IMDB movie review dataset, which consists of 50,000 reviews labeled as either positive or negative.

### Data Preprocessing
Preprocess the text data by performing steps such as removing HTML tags, converting to lowercase, and removing stop words.

*Hint: look at lab **2** for a refresher on Data Preprocessing*

### Tokenization and Padding
Tokenize the text data and pad or truncate the sequences to a fixed length for input to the LSTM network.

```python
max_len = # Fill in  # Maximum review length
X_train = sequence.pad_sequences(# Fill in, maxlen=max_len)
X_test = sequence.pad_sequences(# Fill in, maxlen=max_len)
```

### Word Embeddings
Load pre-trained word embeddings (e.g., GloVe) or train your own word embeddings on the dataset.

### Building the LSTM Model
Build an LSTM model with an embedding layer, LSTM layer(s), and a dense output layer for binary classification (positive/negative).

```python
model = Sequential([
    # Embedding layer: Maps words to dense vectors
    Embedding(# Fill in, # Fill in), 
    # LSTM layer: Captures sequential information
    LSTM(# Fill in),  
    # Output layer: Binary classification (positive/negative)
    Dense(# Fill in, # Fill in)  
])
```

### Training and Evaluation
Train the LSTM model on the training data and evaluate its performance on the testing data using metrics such as accuracy, precision, recall, and F1-score.

```python
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
# Fill in

# Evaluate the model
# Fill in
print(f'Test accuracy: {test_acc:.4f}')
```

### Model Improvement
Explore techniques to improve the model's performance, such as hyperparameter tuning, adding dropout layers, or using pre-trained language models (e.g., BERT).
