import json, os
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchAny,
)

USE_RERANK = False
HISTORY_FILE = "chat_history.json"
COLL = "docs"

def step(n, title):
    print(f"\n=== STEP {n}: {title} ===")

# ---------- Setup ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(":memory:")

docs = [
    {"text": "Qdrant is a vector database used for similarity search.", "source": "qdrant.txt", "access": "public"},
    {"text": "Embeddings turn text into numbers that capture meaning.", "source": "embed.txt", "access": "public"},
    {"text": "Salary details are confidential to the HR team.", "source": "hr.txt", "access": "hr"},
    {"text": "RAG retrieves relevant chunks and gives them to an LLM.", "source": "rag.txt", "access": "public"},
]

client.create_collection(
    COLL, vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
vectors = embedder.encode([d["text"] for d in docs])
client.upsert(
    COLL,
    points=[PointStruct(id=i, vector=vectors[i].tolist(), payload=docs[i])
            for i in range(len(docs))],
)

# ---------- Functions ----------
def validate(question, conv_id):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question is empty")
    if len(question) > 1000:
        raise ValueError("Question too long")
    return question.strip(), conv_id or "default"

def rerank(question, chunks):
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = ce.predict([(question, c["text"]) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked]

def build_prompt(question, chunks):
    context = "\n".join(f"- {c['text']}" for c in chunks)
    return (
        "Answer ONLY using the context below. "
        "If the answer is not there, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

def call_llm(prompt):
    return "[LLM answer will come here]"

def save_history(conv_id, question, answer, sources):
    data = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
    data.setdefault(conv_id, []).append(
        {"question": question, "answer": answer, "sources": sources}
    )
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- Full flow with printing ----------
def ask(question, conv_id=None, user_access=("public",)):
    step(1, "Receive the question")
    print("Question:", question)

    step(2, "Validate input & identify conversation")
    question, conv_id = validate(question, conv_id)
    print("Valid. Conversation ID:", conv_id)

    step(3, "Generate embedding")
    qvec = embedder.encode(question).tolist()
    print("Vector size:", len(qvec))
    print("First 5 numbers:", [round(x, 4) for x in qvec[:5]])

    step(4, "Search vector database")
    step(5, "Apply metadata / access filter")
    print("User access:", list(user_access))
    flt = Filter(must=[FieldCondition(key="access", match=MatchAny(any=list(user_access)))])

    step(6, "Retrieve top matching chunks")
    res = client.query_points(COLL, query=qvec, query_filter=flt, limit=3)
    chunks = [p.payload for p in res.points]
    for p in res.points:
        print(f"  score={p.score:.3f} | {p.payload['source']} | {p.payload['text']}")

    step(7, "Rerank (optional)")
    if USE_RERANK and chunks:
        chunks = rerank(question, chunks)
        print("Reranked order:", [c["source"] for c in chunks])
    else:
        print("Skipped (USE_RERANK = False)")

    step(8, "Build prompt")
    prompt = build_prompt(question, chunks)
    print(prompt)

    step(9, "Send prompt to LLM")
    answer = call_llm(prompt)
    print("LLM replied.")

    step(10, "Return answer + sources")
    sources = [c["source"] for c in chunks]
    print("Answer :", answer)
    print("Sources:", sources)

    step(11, "Store conversation history")
    save_history(conv_id, question, answer, sources)
    print("Saved to", HISTORY_FILE)

    return answer, sources

if __name__ == "__main__":
    ask("What is Qdrant?", conv_id="chat1")
    print("\n" + "=" * 50)
    print("Question Answering Workflow completed successfully!")
    print("=" * 50)