import os
import pandas as pd
from Defense_Scripts.defense_model1 import PhishingDefenseSystem

# Changed from absolute Windows path to relative path
MODEL_PATH = "phishing_model.pkl"

_system = PhishingDefenseSystem()
_MODEL_LOADED = False

def _ensure_model_loaded():
    global _MODEL_LOADED
    if not _MODEL_LOADED:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} not found")
        _system.load_model(MODEL_PATH)
        _MODEL_LOADED = True


def predict_from_excel(
    input_file: str,
    output_file: str,
    sheet_name: str = "Predictions",
    threshold: float = 0.5
):
    _ensure_model_loaded()

    df_test = pd.read_excel(input_file)
    df_test = _system.preprocess(df_test)
    X_test = _system.transform(df_test)

    df_test["phishing_score"] = _system.model.predict_proba(X_test)[:, 1]
    df_test["predicted_label"] = (df_test["phishing_score"] >= threshold).astype(int)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
        df_test[["subject", "sender", "body", "phishing_score", "predicted_label"]] \
            .to_excel(writer, sheet_name=sheet_name, index=False)

    return output_file


def predict_single_email(
    subject: str,
    sender: str,
    body: str,
    threshold: float = 0.5
):
    """
    Predict phishing score and label for a single email.
    Can be imported and used anywhere (API, UI, CLI).
    """
    _ensure_model_loaded()

    email_df = pd.DataFrame([{
        "subject": subject,
        "sender": sender,
        "body": body
    }])

    email_df = _system.preprocess(email_df)
    X_email = _system.transform(email_df)

    phishing_score = _system.model.predict_proba(X_email)[0, 1]
    predicted_label = int(phishing_score >= threshold)

    return {
        "phishing_score": float(phishing_score),
        "predicted_label": predicted_label,
        "prediction": "phishing" if predicted_label == 1 else "legitimate"
    }
