#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hw 4

@author: staciepeng
"""
#%% import data

import pandas as pd
import pickle
import pandas as pd
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

# Load dataset
with open("hw4.pk", "rb") as f:
    data = pickle.load(f)

df = pd.DataFrame(data, columns=["body", "label"])
df = df.rename(columns={"body": "text"})

#%% preprocessing data
# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["clean_text"] = df["text"].apply(clean_text)
#%%
# Run Random Forest with various n-gram settings w/n chi squared
results = []
for ngram in [(1,1), (1,2), (1,3)]:
    vectorizer = CountVectorizer(ngram_range=ngram, stop_words="english", max_features=3000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
   
    results.append({
        "Model": "Random Forest",
        "n-gram": f"{ngram[0]},{ngram[1]}",
        "column": "text",
        "vectorizer": "tf without chi-squared",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fscore": round(fscore, 4)
    })

# Convert to DataFrame
rf_results_df = pd.DataFrame(results)
print(rf_results_df)

#%%
# Run Random Forest with various n-gram settings w chi squared
from sklearn.feature_selection import SelectKBest, chi2
# Initialize results list
results_chi2 = []

# Try multiple n-gram ranges
for ngram in [(1,1), (1,2), (1,3)]:
    # Vectorize
    vectorizer = CountVectorizer(ngram_range=ngram, stop_words="english", max_features=3000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    # Apply Chi-squared feature selection
    chi2_selector = SelectKBest(score_func=chi2, k=1000)
    X_chi2 = chi2_selector.fit_transform(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_chi2, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )

    # Store results
    results_chi2.append({
        "Model": "Random Forest",
        "n-gram": f"{ngram[0]},{ngram[1]}",
        "column": "text",
        "vectorizer": "tf with chi-squared",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fscore": round(fscore, 4)
    })

# Convert to DataFrame and print
rf_chi2_df = pd.DataFrame(results_chi2)
print(rf_chi2_df)
