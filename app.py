from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_DIR = PROJECT_ROOT_DIR / "model" / "faiss_index"
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
SIMILARITY_THRESHOLD = 0.35


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    if not FAISS_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Folder index tidak ditemukan: {FAISS_INDEX_DIR}. Jalankan build_faiss.py dulu."
        )
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=8)
def load_url_documents(url: str):
    loader = WebBaseLoader(url)
    documents = loader.load()

    for document in documents:
        document.metadata["source"] = url
        document.metadata["source_type"] = "url"
        document.metadata["display_source"] = url

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def normalize_score(raw_score: float) -> float:
    """
    Convert FAISS squared L2 distance to a 0..1 cosine-like similarity.
    This works because embeddings are normalized before being indexed.
    """
    similarity = 1.0 - (float(raw_score) / 2.0)
    return max(0.0, min(1.0, similarity))


def search_documents(vectorstore, query: str):
    results = vectorstore.similarity_search_with_score(query, k=TOP_K)
    return [
        {
            "document": document,
            "similarity": normalize_score(score),
            "source": document.metadata.get("display_source")
            or document.metadata.get("source")
            or "Sumber tidak diketahui",
        }
        for document, score in results
    ]


def search_url_documents(query: str, url: str):
    documents = load_url_documents(url)
    if not documents:
        return []

    temp_vectorstore = FAISS.from_documents(documents, get_embeddings())
    results = temp_vectorstore.similarity_search_with_score(query, k=TOP_K)
    return [
        {
            "document": document,
            "similarity": normalize_score(score),
            "source": document.metadata.get("display_source", url),
        }
        for document, score in results
    ]


def combine_results(pdf_results, url_results):
    combined = pdf_results + url_results
    combined.sort(key=lambda item: item["similarity"], reverse=True)
    return combined[:TOP_K]


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.lower()) if len(token) >= 3}


def has_keyword_overlap(question: str, results) -> bool:
    question_tokens = tokenize(question)
    if not question_tokens:
        return False

    for item in results:
        content_tokens = tokenize(item["document"].page_content)
        if question_tokens & content_tokens:
            return True
    return False


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    normalized = clean_text(text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [part.strip() for part in parts if part.strip()]


def extract_title_from_source(source: str) -> str:
    if source.lower().endswith(".pdf"):
        source = source[:-4]
    return clean_text(source.replace("-", " ").replace("_", " "))


def rank_sentences(question: str, content: str) -> list[str]:
    question_tokens = tokenize(question)
    scored = []
    for sentence in split_sentences(content):
        sentence_tokens = tokenize(sentence)
        overlap = len(question_tokens & sentence_tokens)
        bonus = 0.2 if len(sentence) <= 260 else 0.0
        score = overlap + bonus
        if score > 0:
            scored.append((score, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _, sentence in scored]


def build_narrative_answer(question: str, best_result: dict) -> str:
    source = best_result["source"]
    title = extract_title_from_source(source)
    content = clean_text(best_result["document"].page_content)
    ranked_sentences = rank_sentences(question, content)

    if ranked_sentences:
        main_sentence = ranked_sentences[0]
        return f"Berdasarkan artikel \"{title}\", {main_sentence}"

    snippet = content[:400]
    if snippet:
        return f"Berdasarkan artikel \"{title}\", informasi yang paling relevan adalah: {snippet}"

    return "Data tidak ditemukan dalam dokumen"


def generate_answer(question: str, results):
    if not results:
        return "Data tidak ditemukan dalam dokumen"

    best_similarity = results[0]["similarity"]
    if best_similarity < SIMILARITY_THRESHOLD and not has_keyword_overlap(question, results):
        return "Data tidak ditemukan dalam dokumen"

    return build_narrative_answer(question, results[0])


def render_sources(results):
    st.subheader("Explainability")
    for item in results:
        st.write(f"Sumber: {item['source']}")
        st.write(f"Similarity score: {item['similarity']:.4f}")
        st.caption(item["document"].page_content[:300] + "...")


def main():
    st.set_page_config(page_title="Simple R49XAi 4 PDF Chatbot", layout="wide")
    st.title("Simple R49XAi 4 PDF Chatbot")
    st.write("Powered By. AkasSakti")
    
    question = st.text_input("Masukkan pertanyaan")
    url = st.text_input("Masukkan URL opsional")

    if st.button("Kirim", type="primary"):
        if not question.strip():
            st.warning("Pertanyaan wajib diisi.")
            return

        try:
            pdf_results = search_documents(get_vectorstore(), question)
            url_results = search_url_documents(question, url) if url.strip() else []
            final_results = combine_results(pdf_results, url_results)
            answer = generate_answer(question, final_results)
        except Exception as exc:
            st.error(f"Terjadi error: {exc}")
            return

        st.subheader("Jawaban")
        st.write(answer)

        if final_results:
            st.caption(
                f"Best similarity: {final_results[0]['similarity']:.4f} | "
                f"Threshold: {SIMILARITY_THRESHOLD:.2f}"
            )
            render_sources(final_results)


if __name__ == "__main__":
    main()
