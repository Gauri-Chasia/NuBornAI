# config.py — the only file you'll usually need to edit.
# Drop in paths to your own PDFs / .txt / .docx / .html files, or article URLs.

SOURCES = [
    # "/path/to/your/newbornnotes.pdf",
    # "/path/to/your/caring-for-your-newborn-handbook.pdf",
    # "/path/to/your/Healthy Baby Guide BMC.pdf",
    # "/path/to/your/newbornhandbookcoverbooklet.pdf",
]

EMBED_MODEL = "all-mpnet-base-v2"   # sentence-transformers model for dense retrieval
OLLAMA_MODEL = "mistral"            # local model served via `ollama run mistral`

INDEX_PATH = "data/newborn_rag_index.faiss"
META_PATH = "data/newborn_rag_meta.pkl"

FORCE_REBUILD = False   # True = re-parse SOURCES and rebuild the index from scratch
                         # False = load INDEX_PATH / META_PATH from disk if they exist

CHUNK_WORDS = 300
OVERLAP_WORDS = 50

TOP_K = 8     # candidates considered before final trimming
FINAL_K = 5   # chunks actually sent to the model as context
RRF_K = 60    # reciprocal rank fusion constant

# Cross-encoder reranking — a second, more expensive pass that scores
# (question, chunk) pairs directly rather than comparing embeddings, run only
# on the hybrid pool. Meaningfully improves precision over RRF fusion alone.
USE_RERANKER = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 20   # how many hybrid candidates to feed the reranker before trimming to FINAL_K

EVAL_K = 5    # top-k used when scoring retrieval quality in eval/evaluate.py

SYSTEM_PROMPT = """\
You are a warm, careful assistant for new parents of a newborn. Answer using
ONLY the document excerpts provided below.

Rules:
1. If the excerpts contain the answer, explain it clearly and in plain language.
2. Cite the source in brackets, e.g. [some_article.pdf].
3. If the answer is NOT in the excerpts, say so plainly: "I couldn't find this
   in the documents provided — please check with your pediatrician."
4. Never guess or make up medical facts, dosages, or numbers.
5. For anything urgent-sounding (fever, breathing trouble, injury), remind the
   parent this is not a substitute for medical care and to contact their
   pediatrician or emergency services if concerned.
"""
