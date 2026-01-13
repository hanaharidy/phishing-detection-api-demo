import os
import gdown

def ensure_models_exist():
    """Download models if they don't exist locally"""
    
    MODEL1_PATH = "phishing_model.pkl"
    MODEL2_PATH = "modelparameters2.pkl"
    
    model1_exists = os.path.exists(MODEL1_PATH)
    model2_exists = os.path.exists(MODEL2_PATH)
    
    if model1_exists and model2_exists:
        print("✓ Both models already exist locally")
        return
    
    print("📥 Downloading model files from Google Drive...")
    
    # Updated Google Drive file IDs with newly trained models
    MODEL1_URL = "https://drive.google.com/uc?id=1ENnbCXK4bHcKAA0lX20tR-af6jQj3yZF"
    MODEL2_URL = "https://drive.google.com/uc?id=1EaoLcVfqrS7nmz_Khg0RTvQLqPnjzXoQ"
    
    try:
        if not model1_exists:
            print(f"Downloading {MODEL1_PATH}...")
            gdown.download(MODEL1_URL, MODEL1_PATH, quiet=False)
        
        if not model2_exists:
            print(f"Downloading {MODEL2_PATH}...")
            gdown.download(MODEL2_URL, MODEL2_PATH, quiet=False)
        
        print("✅ All models downloaded successfully")
    
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        raise


if __name__ == "__main__":
    ensure_models_exist()
