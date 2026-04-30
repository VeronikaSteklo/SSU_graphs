from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import heapq
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))

from graph import Graph, GraphError
from backend.ml_model import TrafficPredictor

app = FastAPI(title="Traffic Routing API", version="1.0")
# добавить поиск с учетом загруженности дороги
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_graph: Optional[Graph] = None
traffic_model: Optional[TrafficPredictor] = None
latest_traffic_predictions: Optional[dict] = None


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
    global current_graph, traffic_model, latest_traffic_predictions
    filename = payload.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="Не указано имя файла")

    try:
        current_graph = Graph.from_json(filename)
        edges = current_graph.get_edge_list()

        traffic_model = TrafficPredictor()
        traffic_model.train(edges)

        latest_traffic_predictions = None

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
    global latest_traffic_predictions
    if traffic_model is None or current_graph is None:
        raise HTTPException(status_code=404, detail="Модель или граф не инициализированы")

    predictions = traffic_model.predict_all_edges(context.hour, context.day_type, context.weather)
    latest_traffic_predictions = predictions

    return {
        "predictions": [
            {"u": u, "v": v, "base_weight": d['base'], "current_weight": d['current'],
             "congestion_percent": round(d['congestion'], 2)}
            for (u, v), d in predictions.items()
        ],
        "context": context.dict()
    }


def get_effective_graph() -> Graph:
    """Возвращает граф с учетом трафика (если есть предсказания) или базовый граф."""
    if not latest_traffic_predictions:
        return current_graph

    g = Graph.from_copy(current_graph)
    for (u, v), data in latest_traffic_predictions.items():
        try:
            g.change_weight(u, v, data['current'])
            if not g.is_directed and v in g._adj_list and u in g._adj_list[v]:
                g._adj_list[v][u] = data['current']
        except:
            pass
    return g


@app.post("/paths/k-shortest")
async def find_k_shortest_paths(request: PathRequest):
    if current_graph is None:
        raise HTTPException(status_code=404, detail="Граф не создан")
    try:
        effective_graph = get_effective_graph()

        paths = effective_graph.find_k_shortest_paths(request.start, request.end, request.k)

        if not paths: return {"paths": [], "message": "Пути не найдены"}
        return {"paths": [{"index": i, "path": p, "total_weight": round(d, 2),
                           "edges": [(p[j], p[j + 1]) for j in range(len(p) - 1)]}
                          for i, (d, p) in enumerate(paths, 1)]}
    except GraphError as e:
        raise HTTPException(status_code=400, detail=str(e))


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


def _dijkstra_steps(adj_list, start, end):
    steps = []
    distances = {node: float('inf') for node in adj_list}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    edge_to = {}

    steps.append({"message": f"Старт алгоритма Дейкстры. Инициализация: расстояние до {start} = 0", "nodes": [start],
                  "edges": []})

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_node in visited: continue
        visited.add(current_node)

        active_edges = [[edge_to[current_node], current_node]] if current_node in edge_to else []
        steps.append(
            {"message": f"Извлекаем узел {current_node} (расстояние: {current_dist:.2f})", "nodes": [current_node],
             "edges": active_edges})

        if current_node == end:
            steps.append(
                {"message": f"Достигнута целевая вершина {end}!", "nodes": [current_node], "edges": active_edges})
            break

        for neighbor, weight in adj_list.get(current_node, {}).items():
            if neighbor in visited: continue
            steps.append(
                {"message": f"Проверяем соседа {neighbor} (вес {weight:.2f})", "nodes": [current_node, neighbor],
                 "edges": [[current_node, neighbor]]})

            new_dist = current_dist + weight
            if new_dist < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_dist
                edge_to[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))
                steps.append({"message": f"Обновляем расстояние до {neighbor} = {new_dist:.2f}", "nodes": [neighbor],
                              "edges": [[current_node, neighbor]]})

    path = []
    d = distances.get(end, float('inf'))
    if d != float('inf'):
        curr = end
        while curr in edge_to:
            path.append(curr)
            curr = edge_to[curr]
        path.append(start)
        path.reverse()

    res_dist = "inf" if d == float('inf') else round(d, 2)
    result = {"distance": res_dist, "path": path}

    return steps, result


def _kruskal_steps(adj_list):
    steps, edges = [], []
    for u, neighbors in adj_list.items():
        for v, w in neighbors.items():
            if u < v: edges.append((w, u, v))
    edges.sort()

    parent = {n: n for n in adj_list.keys()}

    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    mst_edges = []
    total_weight = 0
    steps.append({"message": "Старт Краскала: все рёбра отсортированы.", "nodes": [], "edges": []})

    for w, u, v in edges:
        steps.append(
            {"message": f"Рассматриваем ребро {u}—{v} (вес {w:.2f})", "nodes": [u, v], "edges": mst_edges + [[u, v]]})
        if union(u, v):
            mst_edges.append([u, v])
            total_weight += w
            steps.append(
                {"message": f"Ребро {u}—{v} добавлено в остовное дерево.", "nodes": [u, v], "edges": list(mst_edges)})
        else:
            steps.append({"message": f"Ребро {u}—{v} отклонено (цикл).", "nodes": [u, v], "edges": list(mst_edges)})

    steps.append({"message": "Алгоритм завершен. MST построено.", "nodes": [], "edges": list(mst_edges)})
    result = {"total_weight": round(total_weight, 2), "mst_edges": mst_edges}
    return steps, result


def _bellman_ford_steps(adj_list, start):
    steps = []
    nodes = list(adj_list.keys())
    dist = {n: float('inf') for n in nodes}
    dist[start] = 0

    edges = [(u, v, w) for u, nbrs in adj_list.items() for v, w in nbrs.items()]
    steps.append({"message": f"Старт Беллмана-Форда. База {start}.", "nodes": [start], "edges": []})

    for i in range(len(nodes) - 1):
        changed = False
        steps.append({"message": f"Итерация {i + 1}.", "nodes": [], "edges": []})
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
                steps.append({"message": f"Релаксация {u}→{v}. Расстояние до {v} = {dist[v]:.2f}", "nodes": [u, v],
                              "edges": [[u, v]]})
        if not changed:
            steps.append({"message": "Досрочное завершение: без релаксаций.", "nodes": [], "edges": []})
            break

    steps.append({"message": "Алгоритм завершен.", "nodes": [], "edges": []})

    dist_str = {k: ("inf" if v == float('inf') else round(v, 2)) for k, v in dist.items()}
    result = {"distances": dist_str}
    return steps, result


@app.post("/paths/visualize-steps/{algo_name}")
async def visualize_algorithm(algo_name: str, request: PathRequest):
    if current_graph is None:
        raise HTTPException(status_code=404, detail="Граф не создан")

    adj = current_graph._adj_list
    start, end = request.start, request.end
    steps, result_data = [], {}

    try:
        if algo_name == "dijkstra":
            steps, result_data = _dijkstra_steps(adj, start, end)
        elif algo_name == "kruskal":
            steps, result_data = _kruskal_steps(adj)
        elif algo_name == "bellman_ford":
            steps, result_data = _bellman_ford_steps(adj, start)
        elif algo_name == "floyd_warshall":
            steps = [{"message": "Флойд-Уоршелл. Матрица рассчитывается...", "nodes": [], "edges": []}]
            result_data = {"message": "Используйте отдельную панель 'Матрица кратчайших путей' для просмотра."}
        elif algo_name == "max_flow":
            steps = [{"message": f"Ищем поток из {start} в {end}", "nodes": [start, end], "edges": []}]
            result_data = {"message": "Поток рассчитан."}
        else:
            raise HTTPException(status_code=400, detail="Алгоритм пока не поддерживается.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации шагов: {str(e)}")

    return {"steps": steps, "result": result_data}


@app.get("/health")
async def health_check():
    return {"status": "ok", "graph_loaded": current_graph is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
