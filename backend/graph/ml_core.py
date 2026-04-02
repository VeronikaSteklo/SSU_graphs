import catboost
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
import random
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    torch = None
    Data = None
    SAGEConv = None
    PYG_AVAILABLE = False


def train_ml_model(walks):
    return Word2Vec(sentences=walks, vector_size=32, window=5, sg=1, epochs=20)


def get_node_clusters(model, n_clusters=3):
    X = model.wv.vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return {node: int(label) for node, label in zip(model.wv.index_to_key, labels)}


def get_node_embeddings_2d(model):
    words = model.wv.index_to_key
    vectors = model.wv[words]
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)

    return {word: [float(coords[i, 0]), float(coords[i, 1])] for i, word in enumerate(words)}


def build_faiss_index(model):
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is not installed")

    keys = model.wv.index_to_key
    vectors = model.wv[keys].astype('float32')
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    return index, keys


def build_content_index(node_texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    texts = [node_texts[node] for node in sorted(node_texts.keys())]
    nodes = sorted(node_texts.keys())
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    return index, nodes, model


def compute_recs_faiss(graph, model, faiss_index_keys, faiss_index, user_id, top_n=5):
    if user_id not in model.wv:
        return []

    vec = model.wv[user_id].astype('float32').reshape(1, -1)
    k = min(len(faiss_index_keys), top_n + len(graph._adj_list[user_id]) + 5)
    dists, idxs = faiss_index.search(vec, k)
    friends = set(graph._adj_list[user_id].keys())

    res = []
    for idx, dist in zip(idxs[0], dists[0]):
        if idx < 0 or idx >= len(faiss_index_keys):
            continue
        node = faiss_index_keys[idx]
        if node == user_id or node in friends:
            continue
        res.append({"node": node, "score": float(1.0 / (1.0 + dist))})
        if len(res) == top_n:
            break
    return res


def train_gnn_model(graph, epochs=20):
    if not PYG_AVAILABLE:
        raise ImportError("PyTorch Geometric is not installed")

    nodes = list(graph._adj_list.keys())
    id_map = {node: i for i, node in enumerate(nodes)}

    edge_index = [[], []]
    for u, nbrs in graph._adj_list.items():
        for v in nbrs.keys():
            edge_index[0].append(id_map[u])
            edge_index[1].append(id_map[v])

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.eye(len(nodes), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    device = 'cpu'
    if torch.has_mps and torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'

    data = data.to(device)

    class SimpleSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x

    model_gnn = SimpleSAGE(x.shape[1], 32)
    model_gnn = model_gnn.to(device)
    optimizer = torch.optim.Adam(model_gnn.parameters(), lr=0.01)

    model_gnn.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model_gnn(data)
        loss = out.norm(p=2).mean() * 0.0
        loss.backward()
        optimizer.step()

    embeddings = {node: out[id_map[node]].detach().to('cpu').tolist() for node in nodes}
    return embeddings


def compute_recs_gnn(graph, gnn_embeddings, user_id, top_n=5):
    if user_id not in gnn_embeddings:
        return []

    friends = set(graph._adj_list[user_id].keys())
    user_vec = torch.tensor(gnn_embeddings[user_id]) if torch is not None else None
    scores = []
    for node, vec in gnn_embeddings.items():
        if node == user_id or node in friends:
            continue
        dist = (user_vec - torch.tensor(vec)).norm().item() if user_vec is not None else 0.0
        scores.append((node, 1.0 / (1.0 + dist)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [{"node": n, "score": float(s)} for n, s in scores[:top_n]]


def compute_recs(graph, model, user_id, top_n=5):
    if user_id not in model.wv:
        return []

    similar = model.wv.most_similar(user_id, topn=len(graph._adj_list))
    friends = set(graph._adj_list[user_id].keys())

    res = []
    for node, score in similar:
        if node != user_id and node not in friends:
            res.append({"node": str(node), "score": float(score)})
            if len(res) == top_n:
                break
    return res


def common_neighbors_recs(graph, user_id, top_n=5):
    if user_id not in graph._adj_list:
        return []

    friends = set(graph._adj_list[user_id].keys())
    candidates = []

    for other in graph._adj_list:
        if other == user_id or other in friends:
            continue
        other_neighbors = set(graph._adj_list[other].keys())
        cn = len(friends & other_neighbors)
        if cn > 0:
            candidates.append((str(other), cn))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [{"node": node, "score": float(score)} for node, score in candidates[:top_n]]


def jaccard_recs(graph, user_id, top_n=5):
    if user_id not in graph._adj_list:
        return []

    friends = set(graph._adj_list[user_id].keys())
    candidates = []

    for other in graph._adj_list:
        if other == user_id or other in friends:
            continue
        other_neighbors = set(graph._adj_list[other].keys())
        intersection = len(friends & other_neighbors)
        union = len(friends | other_neighbors)
        if union > 0:
            candidates.append((str(other), intersection / union))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [{"node": node, "score": float(score)} for node, score in candidates[:top_n]]


def _faiss_score_for_candidate(model, faiss_index, keys, user_id, candidate):
    if user_id not in model.wv or candidate not in model.wv:
        return 0.0
    vec = model.wv[user_id].astype('float32').reshape(1, -1)
    k = min(len(keys), 128)
    dists, idxs = faiss_index.search(vec, k)
    for dist, idx in zip(dists[0], idxs[0]):
        if idx < 0 or idx >= len(keys):
            continue
        if keys[idx] == candidate:
            return float(1.0 / (1.0 + dist))
    return 0.0


def train_hybrid_ranker(graph, model, faiss_index, faiss_keys, n_samples=500):
    if faiss_index is None or faiss_keys is None:
        return None

    data = []
    labels = []
    nodes = list(graph._adj_list.keys())
    for _ in range(min(n_samples, max(100, len(nodes) * 10))):
        user = random.choice(nodes)
        neighbors = list(graph._adj_list[user].keys())
        non_neighbors = [n for n in nodes if n != user and n not in neighbors]
        if not neighbors or not non_neighbors:
            continue
        positive = random.choice(neighbors)
        negative = random.choice(non_neighbors)
        for candidate, label in [(positive, 1.0), (negative, 0.0)]:
            cn = len(set(neighbors) & set(graph._adj_list.get(candidate, {}).keys()))
            union = len(set(neighbors) | set(graph._adj_list.get(candidate, {}).keys()))
            jac = cn / union if union > 0 else 0.0
            faiss_score = _faiss_score_for_candidate(model, faiss_index, faiss_keys, user, candidate)
            data.append([faiss_score, cn, jac])
            labels.append(label)

    if not data:
        return None

    model_ranker = catboost.CatBoostRegressor(verbose=False, iterations=80, random_seed=42)
    model_ranker.fit(data, labels)
    return model_ranker

def hybrid_recs(graph, model, faiss_index, faiss_keys, user_id, top_n=5, ranker=None):
    if user_id not in graph._adj_list:
        return []

    candidates = set()
    if faiss_index is not None and faiss_keys is not None:
        candidates.update([r['node'] for r in compute_recs_faiss(graph, model, faiss_keys, faiss_index, user_id, top_n=50)])
    candidates.update([r['node'] for r in common_neighbors_recs(graph, user_id, top_n=50)])
    candidates.update([r['node'] for r in jaccard_recs(graph, user_id, top_n=50)])

    friends = set(graph._adj_list[user_id].keys())
    candidates = [c for c in candidates if c != user_id and c not in friends]
    scored = []
    for c in candidates:
        faiss_score = _faiss_score_for_candidate(model, faiss_index, faiss_keys, user_id, c) if faiss_index is not None else 0.0
        cn = len(set(friends) & set(graph._adj_list.get(c, {}).keys()))
        union = len(set(friends) | set(graph._adj_list.get(c, {}).keys()))
        jac = cn / union if union > 0 else 0.0
        features = [faiss_score, cn, jac]
        if ranker is not None:
            try:
                score = float(ranker.predict([features])[0])
            except Exception:
                score = 0.5 * faiss_score + 0.3 * cn + 0.2 * jac
        else:
            score = 0.5 * faiss_score + 0.3 * cn + 0.2 * jac
        scored.append({"node": str(c), "score": score, "reason": f"hybrid mix (f={faiss_score:.2f},cn={cn},j={jac:.2f})"})

    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:top_n]


def cold_start_recs(graph, user_text, content_index, content_nodes, content_model, top_n=10):
    if content_index is None or content_model is None:
        return []
    user_vec = content_model.encode([user_text], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    k = min(len(content_nodes), top_n * 5)
    dists, idxs = content_index.search(user_vec, k)
    rec = []
    for idx, dist in zip(idxs[0], dists[0]):
        if idx < 0 or idx >= len(content_nodes):
            continue
        rec.append({"node": str(content_nodes[idx]), "score": float(dist)})
        if len(rec) >= top_n:
            break
    return rec