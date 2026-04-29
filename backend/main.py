from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))

from graph import Graph, GraphError
from backend.ml_model import TrafficPredictor

app = FastAPI(title="Traffic Routing API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_graph: Optional[Graph] = None
traffic_model: Optional[TrafficPredictor] = None


class EdgeUpdate(BaseModel):
    u: str
    v: str
    weight: float


class PathRequest(BaseModel):
    start: str
    end: str
    k: int = 3


class TimeContext(BaseModel):
    hour: int
    day_type: int = 0
    weather: int = 0


class GraphCreate(BaseModel):
    is_directed: bool = False
    is_weighted: bool = True
    vertices: List[str]
    edges: List[Dict]


@app.post("/graph/create")
async def create_graph(data: GraphCreate):
    global current_graph, traffic_model
    try:
        g = Graph(is_directed=data.is_directed, is_weighted=data.is_weighted)
        for v in data.vertices: g.add_vertex(v)
        for edge in data.edges: g.add_edge(edge['u'], edge['v'], edge.get('weight', 1.0))

        current_graph = g
        traffic_model = TrafficPredictor()
        edges_data = [(e['u'], e['v'], e.get('weight', 1.0)) for e in data.edges]
        traffic_model.train(edges_data)

        return {"status": "success", "vertices": len(g._adj_list), "edges": len(g.get_edge_list())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/graphs/list")
async def list_graphs():
    json_files = [f.stem for f in DATA_DIR.glob("*.json")]
    return {"graphs": sorted(json_files)}


@app.post("/graphs/load")
async def load_graph(payload: dict):
    global current_graph, traffic_model
    filename = payload.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="Не указано имя файла")

    try:
        current_graph = Graph.from_json(filename)
        edges = current_graph.get_edge_list()

        traffic_model = TrafficPredictor()
        traffic_model.train(edges)

        return {
            "status": "success",
            "filename": filename,
            "vertices": len(current_graph._adj_list),
            "edges": len(edges)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Файл {filename}.json не найден в data/")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/graph/info")
async def get_graph_info():
    if current_graph is None:
        raise HTTPException(status_code=404, detail="Граф не загружен")
    return {
        "vertices": list(current_graph._adj_list.keys()),
        "edges": current_graph.get_edge_list(),
        "is_directed": current_graph.is_directed,
        "is_weighted": current_graph.is_weighted
    }


@app.post("/traffic/predict")
async def predict_traffic(context: TimeContext):
    if traffic_model is None or current_graph is None:
        raise HTTPException(status_code=404, detail="Модель или граф не инициализированы")

    predictions = traffic_model.predict_all_edges(context.hour, context.day_type, context.weather)

    for (u, v), data in predictions.items():
        try:
            current_graph.change_weight(u, v, data['current'])
            if not current_graph.is_directed and v in current_graph._adj_list and u in current_graph._adj_list[v]:
                current_graph._adj_list[v][u] = data['current']
        except:
            pass

    return {
        "predictions": [
            {"u": u, "v": v, "base_weight": d['base'], "current_weight": d['current'],
             "congestion_percent": round(d['congestion'], 2)}
            for (u, v), d in predictions.items()
        ],
        "context": context.dict()
    }


@app.post("/traffic/update-edge")
async def update_edge_weight(edge: EdgeUpdate):
    if current_graph is None:
        raise HTTPException(status_code=404, detail="Граф не создан")
    try:
        current_graph.change_weight(edge.u, edge.v, edge.weight)
        if not current_graph.is_directed:
            current_graph._adj_list[edge.v][edge.u] = edge.weight
        return {"status": "success", "edge": (edge.u, edge.v), "new_weight": edge.weight}
    except GraphError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/paths/k-shortest")
async def find_k_shortest_paths(request: PathRequest):
    if current_graph is None:
        raise HTTPException(status_code=404, detail="Граф не создан")
    try:
        paths = current_graph.find_k_shortest_paths(request.start, request.end, request.k)
        if not paths: return {"paths": [], "message": "Пути не найдены"}
        return {"paths": [{"index": i, "path": p, "total_weight": round(d, 2),
                           "edges": [(p[j], p[j + 1]) for j in range(len(p) - 1)]}
                          for i, (d, p) in enumerate(paths, 1)]}
    except GraphError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/paths/floyd-warshall")
async def floyd_warshall():
    if current_graph is None:
        raise HTTPException(status_code=404, detail="Граф не создан")
    dist_matrix = current_graph.all_pairs_shortest_paths_floyd()
    nodes = list(current_graph._adj_list.keys())
    return {"nodes": nodes, "matrix": {
        u: {v: ("inf" if dist_matrix[u][v] == float('inf') else round(dist_matrix[u][v], 2)) for v in nodes} for u in
        nodes}}


@app.get("/paths/all-nodes")
async def get_all_nodes():
    if current_graph is None:
        raise HTTPException(status_code=404, detail="Граф не создан")
    return {"nodes": list(current_graph._adj_list.keys())}


@app.get("/visualize/graph")
async def visualize_graph():
    if current_graph is None:
        raise HTTPException(status_code=404, detail="Граф не создан")
    nodes = list(current_graph._adj_list.keys())
    edges = [{"source": u, "target": v, "weight": round(w, 2)} for u, neighbors in current_graph._adj_list.items() for
             v, w in neighbors.items()]
    return {"nodes": nodes, "edges": edges, "is_directed": current_graph.is_directed}


@app.get("/health")
async def health_check():
    return {"status": "ok", "graph_loaded": current_graph is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)