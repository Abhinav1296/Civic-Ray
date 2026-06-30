import nltk

print("⬇️ Downloading NLTK data for Query Expansion...")
try:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    print("✅ NLTK data downloaded successfully!")
except Exception as e:
    print(f"❌ Error downloading NLTK data: {e}")