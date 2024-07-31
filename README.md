# Project README

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction and Visualization](#feature-extraction-and-visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Question Answering with BERT](#question-answering-with-bert)
- [Conclusion and Future Work](#conclusion-and-future-work)

## Overview
This project enhances the accessibility of technical documentation, focusing on the Scikit-Learn user guide. We employ NLP techniques like TF-IDF and BERT to create a centralized system for easy access to complex content.

## Objectives
- **Centralized Documentation**: Unified platform for technical resources.
- **Question Answering**: Precise responses to user queries.
- **Text Classification**: Categorization of documentation sections.

## Data Collection
We initially attempted web scraping, but due to access restrictions, we switched to PDF extraction using PyPDF2.

```python
import requests
from bs4 import BeautifulSoup

def scrape_documentation(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    else:
        return None

documentation_url = "https://scikit-learn.org/stable/modules/linear_model.html"
raw_data = scrape_documentation(documentation_url)
```
## Data Preprocessing

The preprocessing phase involved several critical steps to prepare the text data for analysis:

1. **Cleaning the Text**: We removed HTML tags, special characters, and digits using regular expressions to ensure the data was free from unnecessary elements.

```python
import re

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

cleaned_data = clean_text(raw_data)
```
## Feature Extraction and Visualization

TF-IDF was used for numerical feature extraction, enabling effective text analysis and model training.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform([normalized_data])
feature_names = vectorizer.get_feature_names_out()
```
Visualizations like word clouds and TF-IDF score distributions provided insights into the data.

```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(normalized_data)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```
## Model Training and Evaluation

Logistic Regression and Naive Bayes were used for categorization, leveraging TF-IDF features.
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

documents = [normalized_data]
labels = ['LinearRegression']

X = vectorizer.fit_transform(documents)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
```
## Question Answering with BERT

BERT was utilized for QA tasks, extracting features and predicting answer spans.

```python

from transformers import BertTokenizer, BertForQuestionAnswering

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

def ask_question(question, context):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

context = "Extracted text from the documentation..."
print(ask_question("What is the purpose of the fit method?", context))
```
## Conclusion and Future Work

This project showcases the application of NLP techniques to enhance technical documentation accessibility. Future efforts will focus on expanding to more resources, improving model accuracy, and integrating real-time updates, aligning with SDG 4: Quality Education.

### Contributors
RF Abdullahi & Aisha Kulane
