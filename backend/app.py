from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import List, Dict
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from contextlib import asynccontextmanager
from .graph.graph import generate_social_graph
from .graph.ml_core import (
    train_ml_model, compute_recs, common_neighbors_recs, jaccard_recs,
    get_node_clusters, get_node_embeddings_2d, build_faiss_index, compute_recs_faiss,
    train_gnn_model, compute_recs_gnn, train_hybrid_ranker, hybrid_recs,
    build_content_index, cold_start_recs
)

class RecOut(BaseModel):
    node: str
    score: float
    reason: str | None = None

class GraphDataOut(BaseModel):
    edges: List[Dict]
    clusters: Dict[str, int]


class EmbeddingsOut(BaseModel):
    embeddings: Dict[str, List[float]]


class RecsWithExplanation(BaseModel):
    recs: List[RecOut]
    explanation: str


def generate_recommendation_explanation(user_id: str, recs: List[Dict], method: str):
    if not recs:
        return f"Нет рекомендаций для пользователя {user_id} методом {method}."
    lines = [f"Рекомендации для {user_id} методом {method}:" ]
    for r in recs:
        lines.append(f"{r['node']} (score {r['score']:.3f})")
    lines.append("Причины: модели графовой структуры и сходства эмбеддингов.")
    return " ".join(lines)


class AddEdgeIn(BaseModel):
    source: str
    target: str
    weight: float = 1.0


class ColdStartIn(BaseModel):
    interests: str

data_store = {}


def refresh_models():
    g = data_store["graph"]
    walks = g.get_weighted_random_walks(12, 25)
    model = train_ml_model(walks)
    data_store["model"] = model
    data_store["clusters"] = get_node_clusters(model)
    data_store["embeddings_2d"] = get_node_embeddings_2d(model)

    try:
        idx, keys = build_faiss_index(model)
        data_store["faiss_index"] = idx
        data_store["faiss_keys"] = keys
    except Exception:
        data_store["faiss_index"] = None
        data_store["faiss_keys"] = None

    profile_text = {}
    for node, c in data_store["clusters"].items():
        if c == 0:
            profile_text[node] = "Machine Learning, Python, Data Science"
        elif c == 1:
            profile_text[node] = "Web Development, JavaScript, React"
        else:
            profile_text[node] = "Design, Art, UI/UX"
    try:
        cidx, ckeys, cmodel = build_content_index(profile_text)
        data_store["content_faiss_index"] = cidx
        data_store["content_nodes"] = ckeys
        data_store["content_model"] = cmodel
    except Exception:
        data_store["content_faiss_index"] = None
        data_store["content_nodes"] = None
        data_store["content_model"] = None

    try:
        data_store["gnn_embeddings"] = train_gnn_model(g)
    except Exception:
        data_store["gnn_embeddings"] = None

    try:
        data_store["hybrid_ranker"] = train_hybrid_ranker(g, model, data_store.get("faiss_index"), data_store.get("faiss_keys"))
    except Exception:
        data_store["hybrid_ranker"] = None

    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    g = generate_social_graph(intra_comm_prob=0.4, inter_comm_prob=0.15)
    walks = g.get_weighted_random_walks(12, 25)
    data_store["graph"] = g
    refresh_models()
    yield
    data_store.clear()

app = FastAPI(title="GraphRec Engine", lifespan=lifespan)

@app.get("/graph/data", response_model=GraphDataOut)
async def get_data():
    return {
        "edges": data_store["graph"].get_edge_list_for_api(),
        "clusters": data_store["clusters"]
    }


@app.get("/graph/embeddings")
async def get_embeddings():
    return {
        "embeddings": data_store.get("embeddings_2d", {}),
        "via": "pca2d"
    }


@app.post("/graph/add_edge")
async def add_edge(item: AddEdgeIn):
    g = data_store["graph"]
    if item.source not in g._adj_list or item.target not in g._adj_list:
        raise HTTPException(status_code=404, detail="Node not in graph")
    if item.target in g._adj_list[item.source]:
        raise HTTPException(status_code=400, detail="Edge already exists")
    g.add_edge(item.source, item.target, item.weight)
    return {"status": "ok", "edge": {"source": item.source, "target": item.target, "weight": item.weight}}


@app.post("/graph/remove_edge")
async def remove_edge(item: AddEdgeIn):
    g = data_store["graph"]
    if item.source not in g._adj_list or item.target not in g._adj_list:
        raise HTTPException(status_code=404, detail="Node not in graph")
    if item.target not in g._adj_list[item.source]:
        raise HTTPException(status_code=404, detail="Edge not found")
    g.remove_edge(item.source, item.target)
    refresh_models()
    return {"status": "ok", "removed": {"source": item.source, "target": item.target}}


@app.delete("/graph/nodes/{user_id}")
async def delete_user(user_id: str):
    g = data_store.get("graph")

    if not g or user_id not in g._adj_list:
        raise HTTPException(status_code=404, detail="User not found")

    g.remove_vertex(user_id)

    refresh_models()

    return {"status": "ok", "message": f"User {user_id} deleted and models updated"}


@app.post("/graph/retrain")
async def retrain_graph():
    refresh_models()
    return {"status": "ok", "message": "Models retrained"}

@app.get("/recommend/{user_id}", response_model=RecsWithExplanation)
async def get_recs(user_id: str, method: str = "node2vec", top_n: int = 5):
    if user_id not in data_store["graph"]._adj_list:
        raise HTTPException(status_code=404, detail="User not found")
    if method == "node2vec":
        recs = compute_recs(data_store["graph"], data_store["model"], user_id, top_n=top_n)
    elif method == "common_neighbors":
        recs = common_neighbors_recs(data_store["graph"], user_id, top_n=top_n)
    elif method == "jaccard":
        recs = jaccard_recs(data_store["graph"], user_id, top_n=top_n)
    elif method == "faiss":
        idx = data_store.get("faiss_index")
        keys = data_store.get("faiss_keys")
        if idx is None or keys is None:
            raise HTTPException(status_code=500, detail="FAISS index not available")
        recs = compute_recs_faiss(data_store["graph"], data_store["model"], keys, idx, user_id, top_n=top_n)
    elif method == "gnn":
        gnn_emb = data_store.get("gnn_embeddings")
        if gnn_emb is None:
            raise HTTPException(status_code=500, detail="GNN embeddings not available")
        recs = compute_recs_gnn(data_store["graph"], gnn_emb, user_id, top_n=top_n)
    elif method == "hybrid":
        recs = hybrid_recs(
            data_store["graph"],
            data_store["model"],
            data_store.get("faiss_index"),
            data_store.get("faiss_keys"),
            user_id,
            top_n=top_n,
            ranker=data_store.get("hybrid_ranker")
        )
    else:
        raise HTTPException(status_code=400, detail="Unknown recommend method")
    description_text = generate_recommendation_explanation(user_id, recs, method)
    rec_outs = [RecOut(node=r["node"], score=r["score"], reason=r.get("reason")) for r in recs]
    return {"recs": rec_outs, "explanation": description_text}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/cold_start", response_model=RecsWithExplanation)
async def cold_start(data: ColdStartIn, top_n: int = 10):
    recs = cold_start_recs(
        data_store["graph"],
        data.interests,
        data_store.get("content_faiss_index"),
        data_store.get("content_nodes"),
        data_store.get("content_model"),
        top_n=top_n
    )

    hybrid_ranker = data_store.get("hybrid_ranker")
    if hybrid_ranker is not None and recs:
        nodes = [r["node"] for r in recs]
        hybrid_list = []
        for node in nodes:
            candidate_list = hybrid_recs(
                data_store["graph"], data_store["model"],
                data_store.get("faiss_index"), data_store.get("faiss_keys"),
                node, top_n=1, ranker=hybrid_ranker
            )
            if candidate_list:
                hybrid_list.append(candidate_list[0])
        if hybrid_list:
            recs = hybrid_list[:top_n]
            explanation = "Cold-start candidates reranked by hybrid model"
        else:
            explanation = "Cold-start candidates (hybrid model had no data)"
    else:
        explanation = "Cold-start content-based recommendations"

    rec_outs = [RecOut(node=r["node"], score=r["score"], reason=r.get("reason")) for r in recs]
    return {"recs": rec_outs, "explanation": explanation}


@app.get("/recommend/importance")
async def get_importance():
    ranker = data_store.get("hybrid_ranker")
    if ranker is None:
        raise HTTPException(status_code=404, detail="Hybrid ranker is not available")
    if hasattr(ranker, "get_feature_importance"):
        importances = ranker.get_feature_importance()
        names = ["faiss", "common_neighbors", "jaccard"]
    elif hasattr(ranker, "feature_importances_"):
        importances = ranker.feature_importances_.tolist()
        names = ["faiss", "common_neighbors", "jaccard"]
    else:
        raise HTTPException(status_code=500, detail="Cannot get feature importance")
    return {"feature_importance": dict(zip(names, importances))}