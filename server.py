import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# --- Initialization ---
# Flask app ko initialize karein
app = Flask(__name__)

# .env file se environment variables load karein
load_dotenv()

# Environment variable se apna OpenRouter API key hasil karein
# Yeh key aapki .env file mein honi chahiye
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file. Please add it.")

# Vector store ko save karne ke liye path define karein
VECTOR_PATH = "faiss_index"

# --- Helper Functions (Aapka Original Logic) ---

def process_pdf(filepath):
    """
    PDF se text extract karta hai, chunks banata hai, aur FAISS vector store banakar save karta hai.
    """
    print(f"\nProcessing file: {os.path.basename(filepath)}")
    
    try:
        reader = PdfReader(filepath)
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        
        if not full_text.strip():
            print("Error: Could not extract text from the PDF.")
            return False, "Could not extract text from the PDF. The file might be empty or scanned as an image."

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(full_text)
        print(f"Total text chunks created: {len(chunks)}")
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local(VECTOR_PATH)
        print(f"Vector store saved successfully at: {VECTOR_PATH}")
        return True, f"PDF '{os.path.basename(filepath)}' processed successfully."
        
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        return False, f"An error occurred: {e}"

def get_chat_answer(question):
    """
    Diye gaye sawal ka jawab RAG pipeline ka istemal karke hasil karta hai.
    """
    print(f"\n[USER QUERY] {question}")
    
    try:
        # 1. Vector store load karein
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded.")

        # 2. LLM aur RetrievalQA chain set up karein
        llm = ChatOpenAI(
            model="mistralai/mistral-7b-instruct:free",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.6,
            default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Android RAG App"}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )

        # 3. LLM se jawab hasil karein
        print("Sending query to LLM...")
        result = qa_chain.invoke(question)
        answer = result.get('result', 'Sorry, I could not find an answer.')
        print(f"[FINAL ANSWER] {answer}")
        
        return answer
        
    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        return f"An error occurred while getting the answer: {e}"

# --- API Endpoints ---

@app.route('/process-pdf', methods=['POST'])
def api_process_pdf():
    """
    Yeh API endpoint PDF file ko upload karne aur process karne ke liye hai.
    Request body mein 'file' key ke sath aik PDF file honi chahiye.
    """
    # Check karein ke request mein file hai ya nahi
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # Check karein ke file ka naam hai ya nahi
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        # File ko server par temporarily save karein
        filepath = os.path.join("./", file.filename)
        file.save(filepath)
        
        # PDF ko process karein
        success, message = process_pdf(filepath)
        
        # Temporary file ko delete kardein
        os.remove(filepath)
        
        if success:
            return jsonify({"message": message}), 200
        else:
            return jsonify({"error": message}), 500
            
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400


@app.route('/chat', methods=['POST'])
def api_chat():
    """
    Yeh API endpoint user ke sawal ka jawab dene ke liye hai.
    Request body mein 'question' key ke sath JSON format mein sawal hona chahiye.
    e.g., {"question": "What is this document about?"}
    """
    # Check karein ke vector store mojood hai ya nahi
    if not os.path.exists(VECTOR_PATH):
        return jsonify({"error": "PDF has not been processed yet. Please upload a PDF first."}), 400
        
    # Request se JSON data hasil karein
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided in the request body"}), 400
        
    question = data['question']
    
    # Jawab hasil karein
    answer = get_chat_answer(question)
    
    # Jawab ko JSON format mein wapas bhejein
    return jsonify({"answer": answer})


@app.route('/', methods=['GET'])
def index():
    """
    Aik saada endpoint yeh check karne ke liye ke server chal raha hai.
    """
    return "<h1>PDF Chatbot API</h1><p>Server is running. Use /process-pdf and /chat endpoints.</p>"


# --- Run the App ---
if __name__ == '__main__':
    # 'host="0.0.0.0"' isay network par accessible banata hai
    app.run(host='0.0.0.0', port=5000, debug=True)