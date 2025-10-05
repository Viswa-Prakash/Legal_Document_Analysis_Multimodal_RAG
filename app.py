# app.py
import streamlit as st
from io import BytesIO
from langchain_core.documents import Document
from rag import LegalRAG

st.set_page_config(page_title="Legal Document Assistant", layout="wide")
st.title("⚖️ Legal Document Analysis — Multimodal RAG (PDFs)")

st.markdown("""
Upload legal documents (contracts, agreements, court rulings). The system extracts clauses/sections,
indexes them, and answers legal queries by retrieving relevant clauses and summarizing with GPT-4.1.
""")

# Initialize or reuse RAG instance (cached in Streamlit runtime)
@st.cache_resource
def init_rag():
    # set embed_dim to match your legal model output dimension (e.g., 512 or 768)
    return LegalRAG(embed_dim=512, index_path="faiss_text.index", meta_path="faiss_meta.pkl")

rag = init_rag()

# Sidebar: Upload PDFs + metadata input
st.sidebar.header("📁 Ingest Documents")
uploaded = st.sidebar.file_uploader("Upload PDF files (contracts, rulings, agreements)", type=["pdf"], accept_multiple_files=True)
doc_type = st.sidebar.selectbox("Document type for uploaded files", ["contract", "agreement", "court_ruling", "other"])

if st.sidebar.button("Process & Index Files"):
    if not uploaded:
        st.sidebar.warning("Please select one or more PDF files to ingest.")
    else:
        with st.spinner("Processing PDFs and indexing clauses..."):
            for f in uploaded:
                try:
                    pdf_bytes = f.read()
                    rag.ingest_pdf_bytes(pdf_bytes, source_name=f.name, doc_type=doc_type)
                except Exception as e:
                    st.sidebar.error(f"Failed to ingest {f.name}: {e}")
        st.sidebar.success("Indexing complete. You can now query the corpus.")

# Optionally load existing index
st.sidebar.markdown("---")
if st.sidebar.button("Load saved index (if exists)"):
    try:
        rag.load_index()
        st.sidebar.success("Loaded saved FAISS index & metadata.")
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")

if st.sidebar.button("Save current index"):
    try:
        rag.save_index()
        st.sidebar.success("Saved FAISS index & metadata to disk.")
    except Exception as e:
        st.sidebar.error(f"Save failed: {e}")

# Index stats
st.sidebar.markdown("### Index Info")
try:
    num_indexed = len(rag.vstore.docs)
except Exception:
    num_indexed = 0
st.sidebar.write(f"Indexed chunks: **{num_indexed}**")

# Main query interface
st.header("🔎 Ask a Legal Question")
query_text = st.text_area("Enter your legal question (e.g., 'Is indemnity clause enforceable under state X?')", height=120)

if st.button("Get Legal Answer"):
    if not query_text.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Retrieving relevant clauses and generating answer..."):
            out = rag.query(query_text)
        st.subheader("📌 Answer (from GPT-4.1)")
        st.write(out["text"])

        if out.get("top_hits"):
            st.subheader("📄 Top Matching Clauses / Sections (for context)")
            for i, meta in enumerate(out["top_hits"], start=1):
                src = meta.get("source", "unknown")
                clause = meta.get("clause_number") or meta.get("section_title") or ""
                pg = meta.get("page", "")
                refs = meta.get("legal_references", [])
                st.markdown(f"**{i}. {src} — {clause} (page {pg})**")
                if refs:
                    st.markdown(f"- **Citations found:** {', '.join(refs)}")
                st.markdown("---")

st.markdown("""
---
#### Notes
- Clause/section extraction is heuristic (regex-based) and may need tuning for complex legal formats.
- For production use, consider more robust document parsers (e.g., GROBID) and a dedicated legal embedding model.
- Always review model outputs with legal professionals — this tool assists research but does not replace counsel.
""")
