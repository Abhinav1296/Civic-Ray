import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ⚙️ CONFIGURATION
DATA_PATH = "data"
DB_PATH = "vector_db"
# This model is the "Gold Standard" for accuracy
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def create_vector_db():
    print("🚀 Starting Ingestion Process...")

    # 1️⃣ Load the PDF 📄
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Folder '{DATA_PATH}' missing!")
        return

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print("❌ No PDF found! Please put 'municipal_byelaws.pdf' in data folder.")
        return

    all_docs = []
    for file_name in pdf_files:
        file_path = os.path.join(DATA_PATH, file_name)
        print(f"📖 Reading File: {file_name}...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # --- NEW: Offset Correction Logic ---
        FRONT_MATTER_OFFSET = 17
        for doc in documents:
            # PyPDFLoader populates 'page' in metadata
            absolute_page = doc.metadata.get('page', 0)
            
            # Calculate the logical (printed) page number
            logical_page = absolute_page - FRONT_MATTER_OFFSET
            
            if logical_page > 0:
                doc.metadata['page'] = logical_page
            else:
                doc.metadata['page'] = "Index/Front Matter"
        # ------------------------------------
                
        all_docs.extend(documents)

    # 2️⃣ Split Text into Smart Chunks ✂️
    # We use 1000 characters to keep context intact
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(all_docs)
    print(f"✅ Split into {len(final_docs)} searchable chunks.")

    # 3️⃣ Create the Vector Brain 🧠
    print("🔮 Generating Embeddings (converting text to numbers)...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    db = FAISS.from_documents(final_docs, embeddings)
    db.save_local(DB_PATH)
    print("🎉 Success! Database created at 'vector_db'. Ready for Chat!")

if __name__ == "__main__":
    create_vector_db()