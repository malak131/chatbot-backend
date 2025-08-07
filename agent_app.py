import os
import functools
import gradio as gr
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Make print flush by default (for VSCode)
print = functools.partial(print, flush=True)

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

embedding_model = HuggingFaceEmbeddings()
llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY, model="mistralai/mistral-7b-instruct")

# Define vector index directory
INDEX_DIR = "./indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

# Categories & indexes (add more as needed)
categories = {
    "research": "research_index",
    "legal": "legal_index",
    "medical": "medical_index"
}

retrievers = {}

# ----------- UTILITY FUNCTIONS -----------

def chunk_and_embed(pdf, category):
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    print(f"\n✅ Total Chunks for '{category}': {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"[{i}] {chunk}\n")

    print("🔄 Generating embeddings...")
    vectordb = FAISS.from_texts(chunks, embedding_model)
    index_path = os.path.join(INDEX_DIR, categories[category])
    vectordb.save_local(index_path)
    retrievers[category] = vectordb.as_retriever()

    print(f"✅ Embedding complete. Saved index to: {index_path}\n")

def choose_index(state):
    query = state["query"].lower()
    print(f"\n🔍 Received Query: {query}")

    for category, retriever in retrievers.items():
        if category in query:
            print(f"📌 Routing to index: '{category}'")
            return {"query": query, "retriever": retriever}
    print("⚠️ No specific match found. Defaulting to 'research'")
    return {"query": query, "retriever": retrievers["research"]}

# ----------- AGENT TOOL -----------

@tool
def retrieve_docs(query: str, retriever):
    """Searches the vector store for relevant documents."""
    print("\n🛠️ Retriever Tool Called!")
    docs = retriever.invoke(query)
    print(f"🔎 Retrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs):
        print(f"[Doc {i+1}]\n{doc.page_content}\n")
    return "\n---\n".join([doc.page_content for doc in docs])

# ----------- AGENT SETUP -----------

def create_agent_executor(retriever):
    tool_with_retriever = retrieve_docs.bind(retriever=retriever)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on retrieved knowledge."),
        MessagesPlaceholder("messages"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, [tool_with_retriever], prompt)
    executor = AgentExecutor(agent=agent, tools=[tool_with_retriever], verbose=True)
    return executor

# ----------- LANGGRAPH WORKFLOW -----------

def run_agent(state):
    query = state["query"]
    retriever = state["retriever"]
    executor = create_agent_executor(retriever)
    answer = executor.invoke({"input": query})["output"]
    return {"query": query, "answer": answer}

workflow = StateGraph()
workflow.add_node("router", choose_index)
workflow.add_node("agent", run_agent)
workflow.set_entry_point("router")
workflow.add_edge("router", "agent")
workflow.add_edge("agent", END)
app_graph = workflow.compile()

# ----------- GRADIO UI -----------

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Chat with Multi-Agent PDF Bot")
    gr.Markdown("Upload a PDF, select its category, and ask questions below.")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", type="binary")
        category_input = gr.Radio(choices=list(categories.keys()), label="Select Category", value="research")

    upload_status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(label="Ask a question...")

    def process_pdf(pdf_file, category):
        if pdf_file is None:
            return "⚠️ Please upload a PDF file."

        pdf = PdfReader(pdf_file.name)
        chunk_and_embed(pdf, category)
        return f"✅ PDF processed and index saved for '{category}'."

    def ask_agent(user_msg, history):
        result = app_graph.invoke({"query": user_msg})
        answer = result["answer"]
        history.append((user_msg, answer))
        return "", history

    pdf_input.change(fn=process_pdf, inputs=[pdf_input, category_input], outputs=[upload_status])
    user_input.submit(fn=ask_agent, inputs=[user_input, chatbot], outputs=[user_input, chatbot])

demo.launch()
