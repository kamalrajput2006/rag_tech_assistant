import os
import streamlit as st
import requests

st.set_page_config(page_title="Technical Doc Assistant", layout="centered")
st.title("📚 RAG Technical Documentation Assistant")

API_URL = "http://localhost:8000"

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_user_session"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Detect default sample doc path
default_doc = "data/sample_docs.txt" if os.path.exists("data/sample_docs.txt") else "data/sample_docs.md"

# Sidebar for Ingestion & Settings
with st.sidebar:
    st.header("⚙️ Ingestion Panel")
    source_type = st.selectbox("Source Type", ["file", "url"])
    path_or_url = st.text_input("Path or URL", value=default_doc)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ingest Document", use_container_width=True):
            with st.spinner("Ingesting & updating..."):
                try:
                    res = requests.post(f"{API_URL}/ingest", json={"source_type": source_type, "path_or_url": path_or_url})
                    if res.status_code == 200:
                        st.success(f"Indexed {res.json().get('chunks_indexed')} chunks.")
                        st.rerun()
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Failed to connect: {e}")

    with col2:
        if st.button("Clear DB", use_container_width=True):
            with st.spinner("Clearing..."):
                try:
                    res = requests.post(f"{API_URL}/clear")
                    if res.status_code == 200:
                        st.session_state.messages = []
                        st.warning("Database and chat history cleared!")
                        st.rerun()
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Failed to connect: {e}")

    st.markdown("---")
    st.subheader("📑 Indexed Documents")
    try:
        doc_res = requests.get(f"{API_URL}/documents")
        if doc_res.status_code == 200:
            doc_data = doc_res.json()
            docs = doc_data.get("documents", [])
            total_chunks = doc_data.get("total_chunks", 0)
            st.caption(f"Total chunks stored: **{total_chunks}**")
            if docs:
                for d in docs:
                    st.markdown(f"- `{d}`")
            else:
                st.info("No documents currently indexed.")
    except Exception:
        st.caption("Backend not reached.")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the indexed content..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking & Retrieving..."):
            try:
                res = requests.post(f"{API_URL}/query", json={"question": prompt, "session_id": st.session_state.session_id})
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        st.caption(f"**Sources:** {', '.join(str(s) for s in sources)}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Error communicating with backend API.")
            except Exception as e:
                st.error(f"Connection error: {e}")
