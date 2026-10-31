import streamlit as st
import requests

st.set_page_config(page_title="Technical Doc Assistant", layout="centered")
st.title("📚 RAG Technical Documentation Assistant")

API_URL = "http://localhost:8000"

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_user_session"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Ingestion & Settings
with st.sidebar:
    st.header("⚙️ Ingestion Panel")
    source_type = st.selectbox("Source Type", ["file", "url"])
    path_or_url = st.text_input("Path or URL", value="data/sample_docs.md")
    
    if st.button("Ingest Document"):
        with st.spinner("Ingesting..."):
            res = requests.post(f"{API_URL}/ingest", json={"source_type": source_type, "path_or_url": path_or_url})
            if res.status_code == 200:
                st.success(f"Success! Indexed {res.json().get('chunks_indexed')} chunks.")
            else:
                st.error(res.text)

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a technical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking & Retrieving..."):
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
