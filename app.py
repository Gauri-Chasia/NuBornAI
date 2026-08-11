# app.py — local web server for NewbornAI.
# Run with: python app.py, then open http://127.0.0.1:5050

from flask import Flask, jsonify, render_template, request

import config
from rag_engine import engine

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "ready": engine.ready,
        "error": engine.error,
        "log": engine.status_log,
        "sources": engine.sources_summary() if engine.ready else [],
        "chunk_count": engine.index.ntotal if engine.ready and engine.index else 0,
        "ollama_model": config.OLLAMA_MODEL,
        "reranker_model": config.RERANK_MODEL if engine.reranker is not None else None,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    if not engine.ready:
        return jsonify({"error": "Index isn't ready yet."}), 503

    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Empty message."}), 400

    try:
        result = engine.ask(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    engine.reset()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Loading sources and building/loading the index — this can take a minute...")
    engine.initialize()
    for line in engine.status_log:
        print(line)
    print("\nReady. Open http://127.0.0.1:5050 in your browser.")
    app.run(host="127.0.0.1", port=5050, debug=False)
