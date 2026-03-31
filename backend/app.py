from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager
from .graph.graph import generate_social_graph
from .graph.ml_core import train_ml_model, compute_recs, get_node_clusters

class RecOut(BaseModel):
    node: str
    score: float

class GraphDataOut(BaseModel):
    edges: List[Dict]
    clusters: Dict[str, int]

data_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    g = generate_social_graph()
    walks = g.get_weighted_random_walks(12, 25)
    model = train_ml_model(walks)
    data_store["graph"] = g
    data_store["model"] = model
    data_store["clusters"] = get_node_clusters(model)
    yield
    data_store.clear()

app = FastAPI(title="GraphRec Engine", lifespan=lifespan)

@app.get("/graph/data", response_model=GraphDataOut)
async def get_data():
    return {
        "edges": data_store["graph"].get_edge_list_for_api(),
        "clusters": data_store["clusters"]
    }

@app.get("/recommend/{user_id}", response_model=List[RecOut])
async def get_recs(user_id: str):
    if user_id not in data_store["graph"]._adj_list:
        raise HTTPException(status_code=404, detail="User not found")
    return compute_recs(data_store["graph"], data_store["model"], user_id)