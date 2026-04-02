import streamlit as st
import requests
from streamlit_agraph import agraph, Node, Edge, Config
import altair as alt
import pandas as pd

st.set_page_config(layout="wide", page_title="GraphRec AI")
API_BASE = "http://127.0.0.1:8000"


@st.cache_data(show_spinner="Загрузка графа из API...")
def load_initial_data():
    r = requests.get(f"{API_BASE}/graph/data")
    return r.json()


def refresh_graph_data():
    st.cache_data.clear()
    new_data = load_initial_data()
    st.session_state.graph_data = new_data

    new_user_list = sorted(list(new_data["clusters"].keys()))

    if st.session_state.get("target_user") not in new_user_list:
        st.session_state.target_user = new_user_list[0] if new_user_list else None

    return new_data


st.title("Графовая рекомендательная система")

if "graph_data" not in st.session_state:
    st.session_state.graph_data = load_initial_data()
data = st.session_state.graph_data
clusters = data["clusters"]
edges_raw = data["edges"]

user_list = sorted(list(clusters.keys()))

if "target_user" not in st.session_state or st.session_state.target_user not in user_list:
    st.session_state.target_user = user_list[0] if user_list else None

target_user = st.sidebar.selectbox("Выберите пользователя", user_list, 
                                   index=user_list.index(st.session_state.target_user) if st.session_state.target_user else 0,
                                   key="target_user_select")
if target_user != st.session_state.target_user:
    st.session_state.target_user = target_user

method = st.sidebar.selectbox("Метод рекомендаций", ["node2vec", "hybrid", "faiss", "gnn", "common_neighbors", "jaccard"])
top_n = st.sidebar.slider("Число рекомендаций", 1, 15, 5)

st.sidebar.write("---")
st.sidebar.write("## Cold Start для нового пользователя")
cold_interests = st.sidebar.text_input("Интересы (через запятую)", "Python, ML, нейросети")
if st.sidebar.button("Поиск для нового пользователя"):
    cold_res = requests.post(f"{API_BASE}/cold_start?top_n={top_n}", json={"interests": cold_interests})
    if cold_res.ok:
        cold_payload = cold_res.json()
        cold_recs = cold_payload.get("recs", [])
        cold_explanation = cold_payload.get("explanation", "")
        st.sidebar.write("### Cold Start кандидаты")
        for r in cold_recs:
            st.sidebar.success(f"👤 {r['node']} (score {r['score']:.2f})")
        if cold_explanation:
            st.sidebar.info(f"💬 {cold_explanation}")
    else:
        st.sidebar.error(f"Cold start ошибка: {cold_res.text}")

    st.rerun()

st.sidebar.write("---")

if st.sidebar.button("Найти рекомендации"):
    res = requests.get(f"{API_BASE}/recommend/{target_user}?method={method}&top_n={top_n}")
    rec_payload = res.json()
    recs = rec_payload.get("recs", [])
    explanation = rec_payload.get("explanation", "")

    st.sidebar.write(f"### Топ-{top_n} кандидатов:")
    for r in recs:
        st.sidebar.success(f"👤 {r['node']} (Сходство: {r['score']:.2f})")
    if explanation:
        st.sidebar.info(f"💬 Объяснение: {explanation}")
    recommended_ids = {r['node'] for r in recs}
    importance = {}
    if method == 'hybrid':
        imp_res = requests.get(f"{API_BASE}/recommend/importance")
        if imp_res.ok:
            importance = imp_res.json().get('feature_importance', {})
            st.sidebar.write('### Importance')
            st.sidebar.bar_chart(pd.Series(importance))
    if recs:
        if st.sidebar.button("Добавить первую рекомендованную связь"):
            first = recs[0]
            add_res = requests.post(f"{API_BASE}/graph/add_edge", json={"source": target_user, "target": first['node'], "weight": 1.0})
            if add_res.status_code == 200:
                st.sidebar.success(f"Добавлено ребро {target_user} -> {first['node']}")
                st.session_state.graph_data = refresh_graph_data()
                st.rerun()
            else:
                st.sidebar.error(f"Ошибка добавления: {add_res.text}")
                st.session_state.graph_data = refresh_graph_data()
                st.rerun()
else:
    recs = []
    recommended_ids = set()

st.sidebar.write("---")
st.sidebar.write(f"Модель: {method}, топ {top_n}")

st.sidebar.write("---")
st.sidebar.write("## Graph admin")

admin_user_list = sorted(list(st.session_state.graph_data["clusters"].keys()))

remove_source = st.sidebar.selectbox("Удалить ребро: источник", admin_user_list, key="remove_source_select")
remove_target = st.sidebar.selectbox("Удалить ребро: цель", admin_user_list, key="remove_target_select")
if st.sidebar.button("Удалить ребро"):
    rem = requests.post(f"{API_BASE}/graph/remove_edge", json={"source": remove_source, "target": remove_target, "weight": 1.0})
    if rem.ok:
        st.sidebar.success(f"Ребро {remove_source}->{remove_target} удалено")
        st.session_state.graph_data = refresh_graph_data()
        st.rerun()
    else:
        st.sidebar.error(f"Ошибка: {rem.text}")

remove_user_id = st.sidebar.selectbox("Удалить пользователя", admin_user_list, key="remove_user_select")

if st.sidebar.button("Подтвердить удаление"):
    res = requests.delete(f"{API_BASE}/graph/nodes/{remove_user_id}")

    if res.status_code == 200:
        st.sidebar.success(f"Пользователь {remove_user_id} удален")

        st.cache_data.clear()
        new_data = load_initial_data()
        st.session_state.graph_data = new_data

        new_user_list = sorted(list(new_data["clusters"].keys()))
        if st.session_state.target_user not in new_user_list:
            st.session_state.target_user = new_user_list[0] if new_user_list else None

        st.rerun()
    else:
        st.sidebar.error(f"Ошибка {res.status_code}: {res.text}")

if st.sidebar.button("Переобучить модели сейчас"):
    rt = requests.post(f"{API_BASE}/graph/retrain")
    if rt.ok:
        st.sidebar.success("Models retrained")
        st.session_state.graph_data = refresh_graph_data()
        st.rerun()
    else:
        st.sidebar.error(f"Ошибка: {rt.text}")

emb_res = requests.get(f"{API_BASE}/graph/embeddings")
emb = emb_res.json().get("embeddings", {})

nodes = []
colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A29BFE"]

for n_id, c_id in clusters.items():
    nodes.append(Node(
        id=n_id,
        label=n_id,
        size=25 if n_id == target_user else 15,
        color="#FF6B6B" if n_id == target_user else ("#00CC00" if n_id in recommended_ids else colors[c_id % len(colors)])
    ))
edges = [
    Edge(source=e["source"], target=e["target"], width=e["weight"] / 3)
    for e in edges_raw
    if e["source"] in clusters and e["target"] in clusters
]
for r in recs:
    edges.append(Edge(source=target_user, target=r['node'], color="#FF0000", width=4))

config = Config(width=1000, height=700, physics=True)
agraph(nodes=nodes, edges=edges, config=config)

if emb:
    st.write("## 2D Node2Vec Embeddings")
    import pandas as pd
    emb_df = pd.DataFrame([{"node": k, "x": v[0], "y": v[1], "cluster": clusters.get(k, -1)} for k, v in emb.items()])
    emb_df['label'] = emb_df['node']

    chart = st.altair_chart(
        alt.Chart(emb_df)
           .mark_circle(size=100)
           .encode(
              x='x', y='y', color='cluster:N', tooltip=['node', 'cluster'],
              opacity=alt.condition(alt.datum.node == target_user, alt.value(1.0), alt.value(0.6))
           )
        .interactive(),
        width='stretch'
    )