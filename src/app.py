# app.py
import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Q&A Agent 📚",
    page_icon="🤖",
    layout="wide"
)

# --- App Title and Description ---
st.title("Document Q&A AI Agent 📚🤖")
st.info(
    "Ask questions about your documents or request a search on Arxiv. "
    "The agent will decide which tool to use."
)

# --- API Configuration ---
FASTAPI_URL = "http://127.0.0.1:8000/query"

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ask your question..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send query to backend and get response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                payload = {"query": prompt}
                response = requests.post(FASTAPI_URL, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                # The response from FastAPI is JSON, get the 'answer' field
                answer = response.json().get("answer", "Sorry, something went wrong.")
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.exceptions.RequestException as e:
                error_message = f"Connection to backend failed: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})