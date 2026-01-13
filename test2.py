import os
import pandas as pd
from defense_model2 import EmailClassifier

# Changed from absolute Windows path to relative path
MODEL_FILE = "modelparameters2.pkl"

_classifier = EmailClassifier()
_MODEL_LOADED = False

def _ensure_model_loaded():
    global _MODEL_LOADED
    if not _MODEL_LOADED:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"{MODEL_FILE} not found")
        _classifier.load_model(MODEL_FILE)
        _MODEL_LOADED = True


def predict_single_email(subject: str, sender: str, body: str, threshold: float = 0.5):
    """
    Predict phishing score and label for a single email.
    This function can be imported and used anywhere (API, UI, CLI).
    """
    _ensure_model_loaded()

    email_df = pd.DataFrame([{
        "subject": subject,
        "sender": sender,
        "body": body
    }])

    # Apply same preprocessing used in training
    if hasattr(_classifier, "preprocess"):
        email_df = _classifier.preprocess(email_df)

    if hasattr(_classifier, "transform"):
        X = _classifier.transform(email_df)
    else:
        X = email_df

    phishing_score = _classifier.model.predict_proba(X)[0][1]
    predicted_label = int(phishing_score >= threshold)

    return {
        "phishing_score": round(float(phishing_score), 4),
        "predicted_label": predicted_label,
        "prediction": "phishing" if predicted_label == 1 else "legitimate"
    }
