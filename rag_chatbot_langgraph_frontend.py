"""
Streamlit frontend for the hybrid multimodal RAG chatbot (LangGraph backend).
Run: streamlit run app.py
"""
 
import json
import os
import tempfile
import uuid
 
import streamlit as st
 
from rag_chatbot_langgraph_backend import (
    chatbot,
    evaluate,
    indexed_docs,
    ingest_pdf,
    reset_index,
)
 
st.set_page_config(page_title="Annual Report QA", layout="wide")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
 
 
def new_chat():
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())  # also resets backend memory
 
 
def show_sources(sources, timings):
    if not sources:
        return
    stats = " · ".join(f"{k} {v}" for k, v in (timings or {}).items())
    with st.expander(f"Sources ({len(sources)} pages) · {stats}"):
        for s in sources:
            st.markdown(f"**{s['doc']} - p.{s['page']}**")
            st.text(s["text"][:1500])
 
 
# ===================== SIDEBAR =====================
st.sidebar.title("Annual Report QA")
 
files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
 
if st.sidebar.button("Ingest PDFs"):
    if not files:
        st.sidebar.error("Upload at least one PDF")
    else:
        with st.sidebar:
            for f in files:
                with st.spinner(f"Ingesting {f.name}..."):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(f.getvalue())
                    try:
                        info = ingest_pdf(tmp.name, doc_name=f.name)
                        if info.get("skipped"):
                            st.info(f"{f.name} already indexed")
                        else:
                            st.success(
                                f"{f.name}: {info['pages']} pages, "
                                f"{info['chunks']} chunks, {info['ocr_pages']} OCR"
                            )
                    except Exception as e:
                        st.error(f"{f.name}: {e}")
                    finally:
                        os.remove(tmp.name)
        new_chat()  # fresh chat for the new document set
 
docs = indexed_docs()
selected = (
    st.sidebar.multiselect("Search in", list(docs), default=list(docs)) if docs else []
)
 
use_bm25 = st.sidebar.toggle("Hybrid (BM25 + dense)", value=True)
use_rerank = st.sidebar.toggle("Cross-encoder rerank", value=True)
 
col1, col2 = st.sidebar.columns(2)
if col1.button("Reset chat"):
    new_chat()
    st.rerun()
if col2.button("Clear index"):
    reset_index()
    new_chat()
    st.rerun()
 
# ===================== MAIN =====================
st.title("Chat with your annual reports")
chat_tab, eval_tab = st.tabs(["Chat", "Evaluation"])
 
with chat_tab:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            show_sources(m.get("sources"), m.get("timings"))
 
    question = st.chat_input("Ask something from the reports...", disabled=not docs)
 
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
 
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                out = chatbot(
                    question,
                    thread_id=st.session_state.thread_id,
                    docs=selected or None,
                    use_bm25=use_bm25,
                    use_rerank=use_rerank,
                )
            st.write(out["answer"])
            show_sources(out["sources"], out["timings"])
 
        st.session_state.messages.append({
            "role": "assistant",
            "content": out["answer"],
            "sources": out["sources"],
            "timings": out["timings"],
        })
 
with eval_tab:
    st.caption(
        'Upload a JSON list: [{"question": "...", "answer": "...", "doc": "file.pdf", "page": 12}]. '
        'doc and page are optional. Use answer "NOT IN DOCUMENT" for questions the reports cannot answer.'
    )
    eval_file = st.file_uploader("Eval set (.json)", type=["json"], key="eval_file")
    delay = st.number_input("Delay between questions (s)", 0.0, 30.0, 3.0)  # free-tier rate limits
 
    if st.button("Run evaluation", disabled=not (eval_file and docs)):
        items = json.load(eval_file)
        with st.spinner(f"Running {len(items)} questions..."):
            res = evaluate(
                items,
                delay=delay,
                docs=selected or None,
                use_bm25=use_bm25,
                use_rerank=use_rerank,
            )
        st.metric("Answer accuracy", f"{res['accuracy']:.0%}")
        if res["page_hit_rate"] is not None:
            st.metric("Page hit rate", f"{res['page_hit_rate']:.0%}")
        st.dataframe(res["rows"])
        st.caption("Accuracy is judged by the LLM. Spot-check the rows before quoting a number.")
 