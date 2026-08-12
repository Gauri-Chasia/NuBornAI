# NewbornAI — local web UI

This turns your notebook's RAG pipeline into a small local web app: a Flask
backend running the same parsing/indexing/retrieval logic, with a clean
browser UI instead of ipywidgets. Everything still runs locally — your PDFs
never leave your machine, and answers still come from your local Ollama model.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

You'll also need [Ollama](https://ollama.com) installed and running locally,
with your model pulled, e.g.:

```bash
ollama pull mistral
```

## 2. Add your documents

Open `config.py` and fill in `SOURCES` with paths to your PDFs / .docx / .txt
/ .html files, or article URLs — same as the notebook's Config cell. This is
the only file you should need to edit regularly.

## 3. Run it

```bash
python app.py
```

The first run will parse your documents and build the index (this can take
a minute or two, same as in the notebook — you'll see progress in the
terminal). After that it's cached in `data/`, so future startups are fast
unless you set `FORCE_REBUILD = True` in `config.py`.

Once you see `Ready. Open http://127.0.0.1:5050 in your browser.`, open that
URL.

## What you get

- A chat interface with your question/answer history
- A sidebar listing every document currently indexed, with chunk counts
- Citation chips under each answer showing which source(s) it drew from
- A reset button to clear conversation memory (matches `reset_conversation()`)
- A cross-encoder reranker on top of hybrid search (see below)
- A retrieval evaluation harness (see below)

## Cross-encoder reranking

Dense + BM25 fusion (RRF) is fast but only compares an embedding of the
question to an embedding of each chunk — it never actually reads the two
together. `rag_engine.py` adds a second pass: the top `RERANK_POOL` hybrid
candidates get re-scored by a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2` by default), which takes
`(question, chunk)` as a single input and predicts relevance directly. This
is a real, separately-trained ranking model — not the LLM — so it adds
retrieval quality without adding LLM calls.

Toggle it in `config.py`:

```python
USE_RERANKER = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 20   # candidates handed to the reranker before trimming to FINAL_K
```

`engine.search(query, mode=...)` also supports running any single stage in
isolation — `"dense"`, `"bm25"`, `"hybrid"`, or `"hybrid_rerank"` — which is
what the eval harness uses to compare them.

## Evaluating retrieval quality

`eval/` contains a small ablation study that measures the retriever itself
— Precision@k, Recall@k, MRR, and nDCG@k — across all four pipeline modes.
It never calls Ollama; it only measures which documents get retrieved, not
what the LLM says about them.

1. Fill in `eval/eval_set.json` with real questions about your documents and
   which source file(s) should answer each one:

   ```json
   {
     "question": "How many hours does a newborn sleep at 6 weeks?",
     "relevant_sources": ["Healthy Baby Guide BMC.pdf"]
   }
   ```

   Aim for 20-30 questions spanning different topics for a stable average.

2. Run it:

   ```bash
   python eval/evaluate.py
   ```

   This prints a comparison table (dense vs. bm25 vs. hybrid vs.
   hybrid+rerank) and writes per-question scores to `eval/results.csv`.
   Use `--k` to change the cutoff (default is `EVAL_K` in `config.py`).

## File map

- `config.py` — your sources, model names, chunking/retrieval/reranker settings
- `rag_engine.py` — parsing, chunking, FAISS + BM25 hybrid search, cross-encoder reranking, the ask() pipeline
- `app.py` — Flask server and API routes (`/api/chat`, `/api/status`, `/api/reset`)
- `templates/index.html`, `static/style.css`, `static/app.js` — the UI
- `eval/` — retrieval evaluation harness (`eval_set.json`, `metrics.py`, `evaluate.py`)
- `data/` — cached FAISS index + metadata (auto-created)

## Notes

- If `SOURCES` is empty or a file path isn't found, the app still starts —
  the sidebar will just show no indexed sources until you add valid paths
  and restart.
- To point at a different Ollama model, change `OLLAMA_MODEL` in `config.py`.
- To force a full re-index (e.g. after changing `SOURCES`), set
  `FORCE_REBUILD = True` in `config.py` for one run, then set it back to
  `False`.
