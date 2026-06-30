import nltk
import os

# Set NLTK data path to user directory
nltk_data_path = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

print("⬇️ Downloading NLTK data for Query Expansion...")
try:
    nltk.download('wordnet', download_dir=nltk_data_path)
    nltk.download('omw-1.4', download_dir=nltk_data_path)
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
    print("✅ NLTK data downloaded successfully!")
except Exception as e:
    print(f"❌ Error downloading NLTK data: {e}")
    raise e