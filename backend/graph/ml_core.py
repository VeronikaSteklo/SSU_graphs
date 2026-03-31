from gensim.models import Word2Vec
from sklearn.cluster import KMeans


def train_ml_model(walks):
    return Word2Vec(sentences=walks, vector_size=32, window=5, sg=1, epochs=20)


def get_node_clusters(model, n_clusters=3):
    X = model.wv.vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return {node: int(label) for node, label in zip(model.wv.index_to_key, labels)}


def compute_recs(graph, model, user_id, top_n=5):
    if user_id not in model.wv: return []
    similar = model.wv.most_similar(user_id, topn=len(graph._adj_list))
    friends = graph._adj_list[user_id].keys()

    res = []
    for node, score in similar:
        if node != user_id and node not in friends:
            res.append({"node": node, "score": float(score)})
            if len(res) == top_n: break
    return res