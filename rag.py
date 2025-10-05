# rag.py
import os
import io
import re
import pickle
import numpy as np
import torch
from dotenv import load_dotenv

from PyPDF2 import PdfReader

# Transformers (LegalBERT for text embeddings)
from transformers import AutoTokenizer, AutoModel

# LangChain / LangGraph
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain.schema.messages import HumanMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# FAISS
import faiss

# -----------------------------
# Environment & Device
# -----------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Legal BERT Initialization
# (you can replace with any legal/large model)
# -----------------------------
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-small-uncased"  # example legal model
tokenizer_text = AutoTokenizer.from_pretrained(LEGAL_BERT_MODEL)
model_text = AutoModel.from_pretrained(LEGAL_BERT_MODEL).to(device)
model_text.eval()

# Pydantic config
BaseModel.model_config = {"arbitrary_types_allowed": True}

# -----------------------------
# Embedding Helper (LegalBERT CLS pooling)
# -----------------------------
@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    if not text:
        text = ""
    enc = tokenizer_text(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)
    out = model_text(**enc)
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        pooled = out.last_hidden_state.mean(dim=1)
    pooled = pooled / pooled.norm(dim=-1, keepdim=True)
    return pooled.cpu().numpy()[0].astype(np.float32)

# -----------------------------
# Clause & Reference Extraction Utilities
# -----------------------------
# Regex patterns to detect section/article/clause headers (simple heuristic)
_SECTION_PATTERNS = [
    r"(?m)^\s*(Section|SECTION)\s+(\d+(\.\d+)*)\b",    # Section 1, Section 2.1
    r"(?m)^\s*(Article|ARTICLE)\s+([IVXLC]+)\b",       # Article I, II, III (roman numerals)
    r"(?m)^\s*(Clause|CLAUSE)\s+(\d+)\b",              # Clause 1
    r"(?m)^\s*(\d+\.\d+)\s+",                          # numeric headings 1.1, 2.3
]

# Simple case citation pattern (heuristic)
_CASE_CITATION_RE = re.compile(r"\b[A-Z][a-zA-Z]+ v\. [A-Z][a-zA-Z]+(?:,? \d{4})?\b")

def split_into_sections(text: str):
    """
    Heuristically split `text` into sections/clauses.
    Returns list of dicts: {'title': 'Section 1', 'clause_number': '1', 'text': '...'}
    """
    # Find candidate header indices
    headers = []
    for pat in _SECTION_PATTERNS:
        for m in re.finditer(pat, text):
            headers.append((m.start(), m.group(0).strip()))
    if not headers:
        # fallback: return whole text as single section
        return [{"title": "full_document", "clause_number": None, "text": text.strip()}]

    # Sort headers by position
    headers.sort(key=lambda x: x[0])
    sections = []
    for i, (pos, hdr) in enumerate(headers):
        start = pos
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        sec_text = text[start:end].strip()
        # Extract clause/section number heuristically
        num_match = re.search(r"(\d+(\.\d+)*)", hdr)
        clause_num = num_match.group(1) if num_match else None
        title = hdr.splitlines()[0]
        sections.append({"title": title, "clause_number": clause_num, "text": sec_text})
    return sections

def extract_legal_references(text: str):
    return list(set(m.group(0) for m in _CASE_CITATION_RE.finditer(text)))

# -----------------------------
# FAISS Text-only VectorStore (with persistence)
# -----------------------------
class TextFAISSStore:
    def __init__(self, dimension: int, index_path: str = None, meta_path: str = None):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # normalized vectors => cosine via inner product
        self.docs = []       # list of Document
        self.metadatas = []  # parallel list of metadata dicts
        self.index_path = index_path
        self.meta_path = meta_path

    def add(self, emb: np.ndarray, doc: Document):
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        self.index.add(emb)
        self.docs.append(doc)
        self.metadatas.append(doc.metadata or {})

    def search(self, query_emb: np.ndarray, k: int = 5):
        q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        D, I = self.index.search(q, min(k, len(self.docs)))
        results = []
        for idx in I[0]:
            if idx == -1:
                continue
            results.append(self.docs[idx])
        return results

    def save(self):
        if not self.index_path or not self.meta_path:
            raise ValueError("index_path and meta_path must be set to save.")
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"docs": self.docs, "metadatas": self.metadatas}, f)

    def load(self):
        if not self.index_path or not self.meta_path:
            raise ValueError("index_path and meta_path must be set to load.")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            data = pickle.load(f)
            self.docs = data.get("docs", [])
            self.metadatas = data.get("metadatas", [])

# -----------------------------
# Legal RAG Pipeline
# -----------------------------
class LegalRAG:
    def __init__(self, embed_dim: int = 512, index_path: str = "faiss_text.index", meta_path: str = "faiss_meta.pkl"):
        """
        embed_dim: expected embedding dimension of the chosen model; for small LegalBERT could be 768 or 512.
                   Ensure this matches model_text output dimension.
        index_path/meta_path: optional persistence files.
        """
        self.embed_dim = embed_dim
        self.vstore = TextFAISSStore(dimension=embed_dim, index_path=index_path, meta_path=meta_path)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
        self.llm = init_chat_model("openai:gpt-4.1")
        self.checkpointer = MemorySaver()
        self.graph = self._create_graph()

    # -------------------------
    # PDF ingestion: extract text, split into clauses/sections, attach metadata
    # -------------------------
    def ingest_pdf_bytes(self, pdf_bytes: bytes, source_name: str, doc_type: str = "contract"):
        """
        Extract text from PDF, split into sections/clauses, detect legal references,
        chunk if necessary, embed and add to FAISS.
        """
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text and text.strip():
                pages_text.append((i, text))
            else:
                # If page has no text, try OCR fallback using pdf2image+pytesseract
                try:
                    images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0]) or ""
                        if ocr_text.strip():
                            pages_text.append((i, ocr_text))
                except Exception:
                    # ignore OCR errors per page
                    pass

        if not pages_text:
            return  # nothing to ingest

        # For each page, split into sections/clauses
        for page_num, page_text in pages_text:
            sections = split_into_sections(page_text)
            for sec in sections:
                # Build metadata
                md = {
                    "source": source_name,
                    "page": page_num,
                    "doc_type": doc_type,
                    "clause_number": sec.get("clause_number"),
                    "section_title": sec.get("title"),
                    "legal_references": extract_legal_references(sec.get("text", "")),
                }
                # Use text splitter to further chunk long sections
                base_doc = Document(page_content=sec.get("text", ""), metadata=md)
                chunks = self.splitter.split_documents([base_doc])
                for chunk in chunks:
                    emb = embed_text(chunk.page_content or "")
                    # ensure normalized embedding (embed_text normalizes)
                    doc_for_store = Document(page_content=chunk.page_content, metadata=chunk.metadata)
                    self.vstore.add(emb, doc_for_store)

    # -------------------------
    # LangGraph workflow: retrieve_text -> summarize
    # -------------------------
    def _create_graph(self):
        workflow = StateGraph(dict)

        def retrieve_text(state):
            query_text = state.get("query_text", "")
            if not query_text or not query_text.strip():
                return {"query_text": query_text, "text_hits": []}
            q_emb = embed_text(query_text)
            hits = self.vstore.search(q_emb, k=12)
            # filter to only text-type docs (should be text anyway)
            text_hits = [h for h in hits if (h.metadata or {}).get("doc_type")]
            return {"query_text": query_text, "text_hits": text_hits}

        def summarize(state):
            query_text = state.get("query_text", "")
            text_hits = state.get("text_hits", []) or []

            # Build a concise context list: clause metadata + text (top 6)
            context_blocks = []
            for h in text_hits[:6]:
                meta = h.metadata or {}
                src = meta.get("source", "")
                clause = meta.get("clause_number") or meta.get("section_title") or ""
                header = f"[{src} | {clause} | page:{meta.get('page')}]"
                snippet = h.page_content.strip()
                context_blocks.append(f"{header}\n{snippet}")

            prompt_parts = [
                {"type": "text", "text": f"Legal question: {query_text}\n\n"},
            ]
            if context_blocks:
                prompt_parts.append(
                    {"type": "text", "text": "Relevant extracted clauses/sections:\n" + "\n\n".join(context_blocks) + "\n\n"}
                )
            prompt_parts.append(
                {"type": "text", "text": "Provide a concise legal answer citing which documents and clause numbers you used. If unsure, state the uncertainty and reference the sources used."}
            )

            message = HumanMessage(content=prompt_parts)
            try:
                response = self.llm.invoke([message])
                answer_text = getattr(response, "content", None)
                if answer_text is None:
                    answer_text = str(response)
            except Exception as e:
                answer_text = f"LLM call failed: {e}"

            # Also return the top metadata for display
            top_meta = [h.metadata for h in text_hits[:6]]
            return {"answer": {"text": answer_text, "top_hits": top_meta}}

        workflow.add_node("retrieve_text", retrieve_text)
        workflow.add_node("summarize", summarize)
        workflow.set_entry_point("retrieve_text")
        workflow.add_edge("retrieve_text", "summarize")
        workflow.add_edge("summarize", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # -------------------------
    # Persistence helpers
    # -------------------------
    def save_index(self):
        """Save FAISS index and metadata to disk (paths passed to TextFAISSStore)."""
        self.vstore.save()

    def load_index(self):
        """Load FAISS index and metadata from disk."""
        self.vstore.load()

    # -------------------------
    # Public query interface
    # -------------------------
    def query(self, query_text: str):
        if not query_text or not query_text.strip():
            return {"text": "Please provide a legal query.", "top_hits": []}
        state = {"query_text": query_text}
        res = self.graph.invoke(state, config={"configurable": {"thread_id": "streamlit-session"}})
        ans = res.get("answer") if isinstance(res, dict) else res
        if not ans:
            return {"text": "No answer found.", "top_hits": []}
        return {"text": ans.get("text", ""), "top_hits": ans.get("top_hits", [])}
