import os
import gradio as gr
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# --- SETUP AND CONFIGURATION ---
load_dotenv()
# CORRECTED: Added a fallback value to prevent the API key from being None.
# It will first try to load from the .env file, and if it fails, it will use the hardcoded string.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "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"
VECTOR_PATH = "faiss_index_agent"

AGENT_PROMPT_TEMPLATE = """
You are a helpful and conversational assistant designed to answer questions about a provided document.
You have access to a single tool:
- **search_document**: Use this tool to find information within the user's uploaded PDF.
Your instructions are:
1. For any user question, you **must** use the "search_document" tool to find the most relevant information from the document.
2. Do not rely on your own general knowledge. Your answers must be based only on the content of the PDF.
3. If the document does not contain an answer to the question, clearly state that the information is not available in the provided file.
4. Be friendly and conversational in your responses.

Here is the conversation history:
{chat_history}

Question: {input}

{agent_scratchpad}
"""

# --- PDF PROCESSING AND RAG CHAIN ---
def process_pdf(filepath):
    if not filepath:
        return "Please upload a PDF file first."
    print(f"Processing {os.path.basename(filepath)}...")
    reader = PdfReader(filepath)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    if not full_text.strip():
        return "Error: No text could be extracted from the PDF."
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local(VECTOR_PATH)
        return f"✅ PDF '{os.path.basename(filepath)}' processed. You can now ask questions."
    except Exception as e:
        return f"Error processing PDF: {e}"

def get_rag_chain():
    if not os.path.exists(VECTOR_PATH):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)
    rag_llm = ChatOpenAI(model="mistralai/mistral-7b-instruct:free", openai_api_key=OPENROUTER_API_KEY, openai_api_base="https://openrouter.ai/api/v1", temperature=0.3, default_headers={"HTTP-Referer": "http://localhost", "X-Title": "RAG Chain"})
    return RetrievalQA.from_chain_type(llm=rag_llm, chain_type="stuff", retriever=vector_store.as_retriever())

# --- AGENT SETUP ---
def initialize_agent():
    llm = ChatOpenAI(model="openai/gpt-3.5-turbo", openai_api_key=OPENROUTER_API_KEY, openai_api_base="https://openrouter.ai/api/v1", temperature=0.7, streaming=True, default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Agent"})
    rag_chain = get_rag_chain()
    if not rag_chain:
        return None
    tools = [Tool(name="search_document", func=lambda q: rag_chain.invoke({"query": q}), description="Use this to search for information in the uploaded PDF document.")]
    prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- CHATBOT LOGIC ---
def agent_chat(question, history):
    if not os.path.exists(VECTOR_PATH):
        return "Error: You must upload and process a PDF before asking questions."
    agent_executor = initialize_agent()
    if not agent_executor:
        return "Error: The document search tool could not be initialized."
    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    try:
        response = agent_executor.invoke({"input": question, "chat_history": chat_history_str})
        return response.get('output', "Sorry, I encountered an issue.")
    except Exception as e:
        return f"An error occurred: {e}"

# --- GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Agentic Chat with your PDF")
    gr.Markdown("Upload a PDF, wait for processing, and then have a conversation with an AI agent about its content.")
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", type="filepath", file_types=[".pdf"])
            status_output = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=2):
            gr.ChatInterface(fn=agent_chat, chatbot=gr.Chatbot(height=600, label="Agent Chat", type="messages"), textbox=gr.Textbox(placeholder="Ask the agent a question...", container=False, scale=7))
    pdf_input.upload(fn=process_pdf, inputs=pdf_input, outputs=status_output)

if __name__ == "__main__":
    demo.launch()