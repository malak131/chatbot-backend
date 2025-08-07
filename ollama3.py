import streamlit as st
import os
from dotenv import load_dotenv

# --- LangChain Core Components ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# --- The key component: a stable OpenAI client from LangChain ---
from langchain_openai import ChatOpenAI

# --- Environment Variable Setup ---
load_dotenv()
openrouter_api_key = "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"

# --- Streamlit App Layout ---
st.set_page_config(page_title="Hybrid LangChain + OpenRouter", layout="centered")
st.title("LangChain Demo with an OpenAI-Compatible Client")
st.markdown("""
This app uses the best of both worlds:
- **LangChain:** For `ChatPromptTemplate` and `StrOutputParser`.
- **OpenAI Client:** The stable `ChatOpenAI` class is used to call OpenRouter.
This avoids the previous Pydantic dependency errors.
""")

# --- Check for API Key ---
if not openrouter_api_key:
    st.error("OpenRouter API key not found! Please set OPENROUTER_API_KEY in your .env file.")
    st.stop()

# --- The LangChain Chain Definition ---

# # 1. Prompt Template (from your original code)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful and friendly assistant."),
#         ("user", "Question: {question}")
#     ]
# )

# # 2. LLM Definition (The key change is here)
# # We use the standard ChatOpenAI class but configure it for OpenRouter.
# llm = ChatOpenAI(
#     model="mistralai/mistral-small-3.2-24b-instruct:free",
#     openai_api_key=openrouter_api_key,
#     openai_api_base="https://openrouter.ai/api/v1", # This points the client to OpenRouter
#     temperature=0.7,
#     # Optional: You can still pass OpenRouter-specific headers
#     default_headers={
#         "HTTP-Referer": "http://localhost/streamlit-app",
#         "X-Title": "Streamlit Hybrid Chat",
#     }
# )

# # 3. Output Parser (from your original code)
# output_parser = StrOutputParser()

# # 4. Assembling the chain using LangChain Expression Language (LCEL)
# # This is clean, readable, and fully functional.
# chain = prompt | llm | output_parser


# --- Streamlit User Interface ---
input_text = st.text_input("What do you want to know?")

if input_text:
    with st.spinner("Processing with LangChain and OpenRouter..."):
        try:
            # 1. Prompt Template (from your original code)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful and friendly assistant."),
                    ("user", "Question: {question}")
                ]
            )

            # 2. LLM Definition (The key change is here)
            # We use the standard ChatOpenAI class but configure it for OpenRouter.
            llm = ChatOpenAI(
                model="mistralai/mistral-small-3.2-24b-instruct:free",
                openai_api_key=openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1", # This points the client to OpenRouter
                temperature=0.7,
                # Optional: You can still pass OpenRouter-specific headers
                default_headers={
                    "HTTP-Referer": "http://localhost/streamlit-app",
                    "X-Title": "Streamlit Hybrid Chat",
                }
            )

            # 3. Output Parser (from your original code)
            output_parser = StrOutputParser()

            # 4. Assembling the chain using LangChain Expression Language (LCEL)
            # This is clean, readable, and fully functional.
            chain = prompt | llm | output_parser
            # We invoke the entire chain as a single unit
            response = chain.invoke({"question": input_text})

            st.success("Response Generated:")
            st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Enter a question above to get started.")