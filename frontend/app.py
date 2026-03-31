import streamlit as st
import requests
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(layout="wide", page_title="GraphRec AI")
API_BASE = "http://127.0.0.1:8000"


@st.cache_data(show_spinner="Загрузка графа из API...")
def load_initial_data():
    r = requests.get(f"{API_BASE}/graph/data")
    return r.json()


st.title("Графовая рекомендательная система")

data = load_initial_data()
clusters = data["clusters"]
edges_raw = data["edges"]

user_list = sorted(list(clusters.keys()))
target_user = st.sidebar.selectbox("Выберите пользователя", user_list)

if st.sidebar.button("Найти рекомендации"):
    res = requests.get(f"{API_BASE}/recommend/{target_user}")
    recs = res.json()

    st.sidebar.write("### Топ-5 кандидатов:")
    for r in recs:
        st.sidebar.success(f"👤 {r['node']} (Сходство: {r['score']:.2f})")

nodes = []
colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A29BFE"]

for n_id, c_id in clusters.items():
    nodes.append(Node(
        id=n_id,
        label=n_id,
        size=25 if n_id == target_user else 15,
        color=colors[c_id % len(colors)]
    ))

edges = [Edge(source=e["source"], target=e["target"], width=e["weight"] / 3) for e in edges_raw]

config = Config(width=1000, height=700, physics=True)
agraph(nodes=nodes, edges=edges, config=config)