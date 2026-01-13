from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup
import re

from phishing_evaluator import PhishingEvaluator
from download_models import ensure_models_exist

app = FastAPI(
    title="Ensemble Phishing Detection API",
    version="1.0.0",
    description="Detect phishing emails using an ensemble of ML models"
)

@app.on_event("startup")
def startup_event():
    print("Checking for model files...")
    ensure_models_exist()

evaluator = PhishingEvaluator(threshold=0.6)


class EmailInput(BaseModel):
    subject: str
    sender: str
    body: str

def clean_html(content: str) -> str:
    try:
        soup = BeautifulSoup(content, "lxml")
    except Exception:
        soup = BeautifulSoup(content, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Phishing Detector</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .container {
                max-width: 600px;
                width: 100%;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 24px;
                padding: 50px 40px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                text-align: center;
            }
            
            .logo {
                width: 80px;
                height: 80px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                border-radius: 20px;
                margin: 0 auto 25px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 40px;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            }
            
            h1 {
                font-size: 32px;
                font-weight: 700;
                color: #1a1a1a;
                margin-bottom: 15px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .subtitle {
                font-size: 16px;
                color: #666;
                margin-bottom: 40px;
                line-height: 1.6;
            }
            
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-bottom: 40px;
            }
            
            .feature-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 25px 20px;
                border-radius: 16px;
                transition: transform 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
            }
            
            .feature-icon {
                font-size: 32px;
                margin-bottom: 10px;
            }
            
            .feature-title {
                font-size: 14px;
                font-weight: 600;
                color: #333;
                margin-bottom: 5px;
            }
            
            .feature-desc {
                font-size: 12px;
                color: #666;
            }
            
            .cta-button {
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 18px 50px;
                border-radius: 12px;
                text-decoration: none;
                font-weight: 600;
                font-size: 18px;
                transition: all 0.3s ease;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            }
            
            .cta-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
            }
            
            .api-info {
                margin-top: 40px;
                padding-top: 30px;
                border-top: 2px solid #e0e0e0;
            }
            
            .api-endpoint {
                background: #f5f7fa;
                padding: 12px 20px;
                border-radius: 8px;
                font-family: 'Monaco', monospace;
                font-size: 14px;
                color: #667eea;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">🛡️</div>
            <h1>AI Phishing Detector</h1>
            <p class="subtitle">
                Dual-model ensemble system that analyzes emails using advanced machine learning
                to protect you from phishing attacks
            </p>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-title">Dual Models</div>
                    <div class="feature-desc">Two ML models working together</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-title">Real-time</div>
                    <div class="feature-desc">Instant threat detection</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-title">High Accuracy</div>
                    <div class="feature-desc">Maximum score classification</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔒</div>
                    <div class="feature-title">Secure</div>
                    <div class="feature-desc">Privacy-focused analysis</div>
                </div>
            </div>
            
            <a href="/predict_form" class="cta-button">Try It Now →</a>
            
            <div class="api-info">
                <div style="font-size: 14px; color: #666; margin-bottom: 10px;">
                    <strong>API Endpoints:</strong>
                </div>
                <div class="api-endpoint">POST /predict</div>
                <div class="api-endpoint">GET /predict_form</div>
            </div>
        </div>
    </body>
    </html>
    """

@app.post("/predict")
def predict_email(data: EmailInput):
    try:
        cleaned_body = clean_html(data.body)

        result = evaluator.classify_single_email(
            subject=data.subject,
            sender=data.sender,
            body=cleaned_body
        )

        return {
            "subject": data.subject,
            "sender": data.sender,
            "cleaned_body": cleaned_body,
            **result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict_form", response_class=HTMLResponse)
def predict_form():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email Phishing Analyzer</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 40px 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 24px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            
            .back-link {
                display: inline-block;
                color: #667eea;
                text-decoration: none;
                font-size: 14px;
                margin-bottom: 20px;
                transition: opacity 0.3s;
            }
            
            .back-link:hover {
                opacity: 0.7;
            }
            
            h2 {
                font-size: 28px;
                color: #1a1a1a;
                margin-bottom: 10px;
            }
            
            .subtitle {
                color: #666;
                font-size: 14px;
            }
            
            .form-group {
                margin-bottom: 25px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
                font-size: 14px;
            }
            
            input[type="text"], textarea {
                width: 100%;
                padding: 14px 18px;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                font-size: 15px;
                font-family: inherit;
                transition: all 0.3s ease;
                background: #fafafa;
            }
            
            input[type="text"]:focus, textarea:focus {
                outline: none;
                border-color: #667eea;
                background: white;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            textarea {
                resize: vertical;
                min-height: 150px;
            }
            
            .submit-btn {
                width: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            }
            
            .submit-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
            }
            
            .submit-btn:active {
                transform: translateY(0);
            }
            
            .info-box {
                background: linear-gradient(135deg, #e0f7fa 0%, #e1bee7 100%);
                padding: 20px;
                border-radius: 12px;
                margin-top: 30px;
                font-size: 13px;
                color: #555;
            }
            
            .info-box strong {
                color: #333;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <a href="/" class="back-link">← Back to Home</a>
                <h2>🔍 Analyze Email</h2>
                <p class="subtitle">Paste the email details below to check if it's a phishing attempt</p>
            </div>
            
            <form action="/predict_form" method="post">
                <div class="form-group">
                    <label>📧 Email Subject</label>
                    <input type="text" name="subject" placeholder="e.g., Urgent: Verify your account" required>
                </div>
                
                <div class="form-group">
                    <label>👤 Sender Email</label>
                    <input type="text" name="sender" placeholder="e.g., security@suspicious-site.com" required>
                </div>
                
                <div class="form-group">
                    <label>📄 Email Body</label>
                    <textarea name="body" placeholder="Paste the full email content here..." required></textarea>
                </div>
                
                <button type="submit" class="submit-btn">Analyze Email 🔍</button>
            </form>
            
            <div class="info-box">
                <strong>How it works:</strong> Our dual-model AI system analyzes the subject, sender, and body of the email using two independent machine learning models. The final verdict uses the maximum phishing score from both models to ensure maximum accuracy.
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/predict_form", response_class=HTMLResponse)
def predict_form_post(
    subject: str = Form(...),
    sender: str = Form(...),
    body: str = Form(...)
):
    try:
        cleaned_body = clean_html(body)

        result = evaluator.classify_single_email(
            subject=subject,
            sender=sender,
            body=cleaned_body
        )

        is_phishing = result['result'] == "PHISHING"
        color = "#e53e3e" if is_phishing else "#38a169"
        bg_color = "#fff5f5" if is_phishing else "#f0fff4"
        icon = "⚠️" if is_phishing else "✅"
        
        score_display_1 = result['phishing_score_model1']
        score_display_2 = result['phishing_score_model2']
        max_score_display = result['max_phishing_score']

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Analysis Result</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 40px 20px;
                }}
                
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 24px;
                    padding: 40px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                }}
                
                .back-link {{
                    display: inline-block;
                    color: #667eea;
                    text-decoration: none;
                    font-size: 14px;
                    margin-bottom: 30px;
                    transition: opacity 0.3s;
                }}
                
                .back-link:hover {{
                    opacity: 0.7;
                }}
                
                .result-badge {{
                    text-align: center;
                    padding: 30px;
                    background: {bg_color};
                    border: 3px solid {color};
                    border-radius: 20px;
                    margin-bottom: 40px;
                }}
                
                .result-icon {{
                    font-size: 60px;
                    margin-bottom: 15px;
                }}
                
                .result-text {{
                    font-size: 32px;
                    font-weight: 700;
                    color: {color};
                    margin-bottom: 10px;
                }}
                
                .result-subtitle {{
                    font-size: 16px;
                    color: #666;
                }}
                
                .scores-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                
                .score-card {{
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 25px;
                    border-radius: 16px;
                    text-align: center;
                }}
                
                .score-label {{
                    font-size: 12px;
                    color: #666;
                    margin-bottom: 10px;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                
                .score-value {{
                    font-size: 36px;
                    font-weight: 700;
                    color: #1a1a1a;
                }}
                
                .score-bar {{
                    width: 100%;
                    height: 8px;
                    background: #e0e0e0;
                    border-radius: 4px;
                    margin-top: 10px;
                    overflow: hidden;
                }}
                
                .score-fill {{
                    height: 100%;
                    background: linear-gradient(90deg, #667eea, #764ba2);
                    border-radius: 4px;
                    transition: width 0.5s ease;
                }}
                
                .email-details {{
                    background: #fafafa;
                    padding: 25px;
                    border-radius: 16px;
                    margin-bottom: 30px;
                }}
                
                .detail-row {{
                    margin-bottom: 20px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #e0e0e0;
                }}
                
                .detail-row:last-child {{
                    margin-bottom: 0;
                    padding-bottom: 0;
                    border-bottom: none;
                }}
                
                .detail-label {{
                    font-size: 12px;
                    font-weight: 600;
                    color: #666;
                    margin-bottom: 8px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                
                .detail-value {{
                    font-size: 14px;
                    color: #333;
                    line-height: 1.6;
                    word-break: break-word;
                }}
                
                .action-button {{
                    display: inline-block;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 14px 40px;
                    border-radius: 12px;
                    text-decoration: none;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
                }}
                
                .action-button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
                }}
                
                .center {{
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/predict_form" class="back-link">← Analyze Another Email</a>
                
                <div class="result-badge">
                    <div class="result-icon">{icon}</div>
                    <div class="result-text">{result['result']}</div>
                    <div class="result-subtitle">
                        {'This email shows signs of phishing. Be cautious!' if is_phishing else 'This email appears to be legitimate.'}
                    </div>
                </div>
                
                <div class="scores-grid">
                    <div class="score-card">
                        <div class="score-label">Model 1 Score</div>
                        <div class="score-value">{score_percent_1}%</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {score_percent_1}%"></div>
                        </div>
                    </div>
                    
                    <div class="score-card">
                        <div class="score-label">Model 2 Score</div>
                        <div class="score-value">{score_percent_2}%</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {score_percent_2}%"></div>
                        </div>
                    </div>
                    
                    <div class="score-card">
                        <div class="score-label">Final Score (Max)</div>
                        <div class="score-value">{max_score_percent}%</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {max_score_percent}%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="email-details">
                    <div class="detail-row">
                        <div class="detail-label">📧 Subject</div>
                        <div class="detail-value">{subject}</div>
                    </div>
                    
                    <div class="detail-row">
                        <div class="detail-label">👤 Sender</div>
                        <div class="detail-value">{sender}</div>
                    </div>
                    
                    <div class="detail-row">
                        <div class="detail-label">📄 Email Content (Cleaned)</div>
                        <div class="detail-value">{cleaned_body[:500]}{'...' if len(cleaned_body) > 500 else ''}</div>
                    </div>
                </div>
                
                <div class="center">
                    <a href="/predict_form" class="action-button">Analyze Another Email</a>
                </div>
            </div>
        </body>
        </html>
        """
        return html

    except Exception as e:
        return f"<p style='color:red'>Error: {str(e)}</p>"