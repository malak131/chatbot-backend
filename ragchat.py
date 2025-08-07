import os
import gradio as gr
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "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"
VECTOR_PATH = "faiss_index"

def process_pdf(filepath):
    """Extracts text from a PDF, creates a vector store, and returns a status message."""
    if filepath is None:
        return "Please upload a PDF file first."
    
    print(f"\nProcessing file: {os.path.basename(filepath)}")
    
    reader = PdfReader(filepath)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    print(f"Total text chunks created: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"[CHUNK {i+1}] {chunk[:200]}{'...' if len(chunk) > 200 else ''}")  # printing preview

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    print(f" Embeddings created for chunks. Example vector size: {len(vector_store.index.reconstruct(0))}")

    vector_store.save_local(VECTOR_PATH)
    print(f"Vector store saved at: {VECTOR_PATH}")
    
    return f"PDF '{os.path.basename(filepath)}' processed. You can now ask questions."

def chat_with_bot(question, history):
    """Answers a question using the RAG pipeline. Handles the case where no PDF is processed."""
    if not os.path.exists(VECTOR_PATH):
        return "Error: You must upload and process a PDF before asking questions."
        
    print(f"\n[USER QUERY] {question}")
    
    # 1. Load the vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)

    print("Vector store loaded.")

    # 2. Set up the LLM and RetrievalQA chain
    llm = ChatOpenAI(
        model="mistralai/mistral-7b-instruct:free",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.6,
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Gradio RAG"}
    )
    
    retriever = vector_store.as_retriever()
    
    print("Fetching relevant chunks for the query...")
    relevant_docs = retriever.get_relevant_documents(question)
    
    print(f" {len(relevant_docs)} relevant chunks found:")
    for i, doc in enumerate(relevant_docs):
        print(f"[MATCHED CHUNK {i+1}] {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    print("Sending query to LLM...")
    result = qa_chain.invoke(question)

    answer = result.get('result', 'Sorry, I could not find an answer.')
    print(f"[FINAL ANSWER]{answer}")
    
    return answer

# --- GRADIO UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Chat with your PDF")
    gr.Markdown("Upload a PDF, wait for the status message, and then ask questions below.")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", type="filepath", file_types=[".pdf"])
            status_output = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            gr.ChatInterface(
                fn=chat_with_bot,
                chatbot=gr.Chatbot(height=500, label="Chat", type="messages"),
                textbox=gr.Textbox(placeholder="Ask a question about your document...", container=False, scale=7),
            )

    pdf_input.upload(fn=process_pdf, inputs=pdf_input, outputs=status_output)

if __name__ == "__main__":
    demo.launch()
