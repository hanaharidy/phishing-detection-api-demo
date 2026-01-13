import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
import joblib
import os
# from ydata_profiling import ProfileReport
class PhishingDefenseSystem:
    """
    End-to-end AI defense system for phishing detection
    """

    def __init__(self):
        self.tfidf_subject = TfidfVectorizer(
            ngram_range=(1, 3), max_features=4000, stop_words="english", sublinear_tf=True
        )
        self.tfidf_sender = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=2000)
        self.tfidf_body = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=12000, sublinear_tf=True)

        lr = LogisticRegression(max_iter=3000, class_weight="balanced")
        nb = MultinomialNB()
        svm = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="sigmoid")

        self.model = VotingClassifier(estimators=[("lr", lr), ("nb", nb), ("svm", svm)], voting="soft")

    @staticmethod
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "<URL>", text)
        text = re.sub(r"\S+@\S+", "<EMAIL>", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def preprocess(self, df):
        df["subject_clean"] = df["subject"].apply(self.clean_text)
        df["sender_clean"] = df["sender"].apply(self.clean_text)
        df["body_clean"] = df["body"].apply(self.clean_text)
        return df

    def fit_transform(self, df):
        X_sub = self.tfidf_subject.fit_transform(df["subject_clean"])
        X_send = self.tfidf_sender.fit_transform(df["sender_clean"])
        X_body = self.tfidf_body.fit_transform(df["body_clean"])
        return hstack([X_sub, X_send, X_body])

    def transform(self, df):
        X_sub = self.tfidf_subject.transform(df["subject_clean"])
        X_send = self.tfidf_sender.transform(df["sender_clean"])
        X_body = self.tfidf_body.transform(df["body_clean"])
        return hstack([X_sub, X_send, X_body])

   
    def save_model(self, model_file="phishing_model.pkl"):
        """
        Save the model and TF-IDF transformers locally.
        This will create a new file in the folder where the script is running.
        """
        # Get current folder
        current_folder = os.getcwd()  # folder where the script is running
        full_path = os.path.join(current_folder, model_file)

        # Save model and transformers
        joblib.dump({
            "model": self.model,
            "tfidf_subject": self.tfidf_subject,
            "tfidf_sender": self.tfidf_sender,
            "tfidf_body": self.tfidf_body
        }, full_path)

    def load_model(self, model_file="phishing_model1_parameters.pkl"):
        data = joblib.load(model_file)
        self.model = data["model"]
        self.tfidf_subject = data["tfidf_subject"]
        self.tfidf_sender = data["tfidf_sender"]
        self.tfidf_body = data["tfidf_body"]
        print(f" Model and transformers loaded from {model_file}")
