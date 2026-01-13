import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split

class EmailClassifier:
    """Email phishing classifier with file I/O like PhishingDefenseSystem"""
    def __init__(self, max_iter=500):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("sender", OneHotEncoder(handle_unknown='ignore'), ["sender"]),
                ("subject_tfidf", TfidfVectorizer(stop_words="english", max_features=3000), "subject"),
                ("body_tfidf", TfidfVectorizer(stop_words="english", max_features=5000), "body"),
            ],
            remainder="drop"
        )
        self.model = Pipeline([
            ("prep", self.preprocessor),
            ("clf", LogisticRegression(max_iter=max_iter))
        ])

    def load_data(self, filepath, test_size=0.2):
        df = pd.read_excel(filepath) 
        df = df.drop_duplicates().fillna("")
        X = df[["sender", "subject", "body"]]
        y = df["label"]
        return train_test_split(X, y, test_size=test_size, random_state=42)


    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print(" Model trained successfully")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model.named_steps["clf"], "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        print(classification_report(y_test, y_pred, digits=4))
        print(f"Accuracy:  {acc*100:.1f}%")
        print(f"Precision: {prec*100:.1f}%")
        print(f"Recall:    {rec*100:.1f}%")
        print(f"F1-Score:  {f1*100:.1f}%")
        if auc is not None:
            print(f"ROC-AUC:   {auc:.3f}")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

    def save_model(self, model_file="email_model.pkl"):
        current_folder = os.getcwd()
        full_path = os.path.join(current_folder, model_file)
        joblib.dump(self.model, full_path)
        print(f"Model saved to {full_path}")

    def load_model(self, model_file="email_model.pkl"):
        current_folder = os.getcwd()
        full_path = os.path.join(current_folder, model_file)
        self.model = joblib.load(full_path)
        print(f" Model loaded from {full_path}")

    def predict_single(self, sender, subject, body):
        X_new = pd.DataFrame([[sender, subject, body]], columns=["sender", "subject", "body"])
        X_new = X_new.fillna("")  
        label = self.model.predict(X_new)[0]
        proba = self.model.predict_proba(X_new)[:, 1][0] if hasattr(self.model.named_steps["clf"], "predict_proba") else None
        return label, proba


    def predict_to_excel(self, input_file, output_file, sheet_name="Predictions"):
        df_test = pd.read_excel(input_file)
        df_test["predicted_label"] = ""
        df_test["phishing_score"] = 0.0

        for i, row in df_test.iterrows():
            label, score = self.predict_single(row["sender"], row["subject"], row["body"])
            df_test.at[i, "predicted_label"] = label
            df_test.at[i, "phishing_score"] = score if score is not None else 0.0

        if os.path.exists(output_file):
            with pd.ExcelWriter(output_file, engine="openpyxl", mode="a") as writer:
                df_test.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
                df_test.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f" Predictions saved in sheet '{sheet_name}' of {output_file}")
