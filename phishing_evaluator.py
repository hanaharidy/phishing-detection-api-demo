import pandas as pd
from test1 import predict_single_email as predict_model1
from test2 import predict_single_email as predict_model2

class PhishingEvaluator:
    """
    Takes an email (or DataFrame of emails), predicts phishing
    using two models, and uses the MAX score for final classification.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def classify_single_email(self, subject: str, sender: str, body: str) -> dict:
        """
        Predict phishing for a single email using BOTH models
        and return the max score and final label.
        """

        try:
            # Model 1
            result1 = predict_model1(subject, sender, body, self.threshold)
            score1 = result1["phishing_score"]

            # Model 2
            result2 = predict_model2(subject, sender, body, self.threshold)
            score2 = result2["phishing_score"]

            # Take max score
            max_score = max(score1, score2)
            final_label = "PHISHING" if max_score >= self.threshold else "SAFE"

            return {
                "phishing_score_model1": score1,
                "phishing_score_model2": score2,
                "max_phishing_score": round(max_score, 4),
                "result": final_label
            }
        except FileNotFoundError as e:
            # Friendly error if model files missing
            return {
                "phishing_score_model1": None,
                "phishing_score_model2": None,
                "max_phishing_score": None,
                "result": f"Error: {str(e)}"
            }

    def classify_dataframe(self, df: pd.DataFrame, subject_col="subject", sender_col="sender", body_col="body") -> pd.DataFrame:
        """
        Classify a DataFrame of emails using both models and MAX score.
        Returns a DataFrame with scores and results.
        """
        records = []

        for _, row in df.iterrows():
            result = self.classify_single_email(
                subject=row[subject_col],
                sender=row[sender_col],
                body=row[body_col]
            )
            records.append({
                subject_col: row[subject_col],
                sender_col: row[sender_col],
                body_col: row[body_col],
                **result
            })

        return pd.DataFrame(records)
