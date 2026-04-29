import streamlit as st
import requests
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Smart Traffic Routing",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"


def get_graph_info():
    try:
        response = requests.get(f"{API_URL}/graph/info", timeout=2)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
    except requests.exceptions.RequestException as e:
        st.warning(f"Нет связи с бэкендом: {e}")
        return None
    return None


def predict_traffic(hour, day_type, weather):
    try:
        response = requests.post(
            f"{API_URL}/traffic/predict",
            json={"hour": hour, "day_type": day_type, "weather": weather}
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")
    return None


def find_paths(start, end, k=3):
    try:
        response = requests.post(
            f"{API_URL}/paths/k-shortest",
            json={"start": start, "end": end, "k": k}
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Ошибка поиска путей: {e}")
    return None


def get_visualization_data():
    """Получение данных для визуализации"""
    try:
        response = requests.get(f"{API_URL}/visualize/graph")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def create_graph_visualization(viz_data, paths=None, congestion_data=None):
    """Создание интерактивной визуализации графа с группировкой рёбер по цвету"""
    if not viz_data:
        return None

    G = nx.DiGraph() if viz_data.get('is_directed', False) else nx.Graph()
    G.add_nodes_from(viz_data['nodes'])

    edge_weights = {}
    edge_congestion = {}

    for edge in viz_data['edges']:
        u, v = edge['source'], edge['target']
        weight = edge['weight']
        G.add_edge(u, v, weight=weight)
        edge_weights[(u, v)] = weight

        if congestion_data:
            for pred in congestion_data.get('predictions', []):
                if (pred['u'] == u and pred['v'] == v) or (pred['u'] == v and pred['v'] == u):
                    edge_congestion[(u, v)] = pred['congestion_percent']
                    break

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    path_edges = set()
    if paths and paths.get('paths'):
        for path_info in paths['paths']:
            path = path_info['path']
            for i in range(len(path) - 1):
                path_edges.add((path[i], path[i + 1]))

    edge_groups = {'path': [], 'high': [], 'medium': [], 'low': []}

    for edge in G.edges():
        congestion = edge_congestion.get(edge, 0)

        if edge in path_edges:
            edge_groups['path'].append(edge)
        elif congestion > 50:
            edge_groups['high'].append(edge)
        elif congestion > 20:
            edge_groups['medium'].append(edge)
        else:
            edge_groups['low'].append(edge)

    styles = {
        'path': {'color': 'rgb(255, 0, 0)', 'width': 4, 'name': 'Выбранный путь'},
        'high': {'color': 'rgb(255, 100, 100)', 'width': 3, 'name': 'Высокая загрузка'},
        'medium': {'color': 'rgb(255, 200, 100)', 'width': 2.5, 'name': 'Средняя загрузка'},
        'low': {'color': 'rgb(100, 100, 200)', 'width': 2, 'name': 'Низкая загрузка'}
    }

    edge_traces = []
    for group_key, edges in edge_groups.items():
        if not edges:
            continue

        style = styles[group_key]
        edge_x, edge_y, edge_texts = [], [], []

        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            weight = edge_weights.get(edge, 1.0)
            congestion = edge_congestion.get(edge, 0)
            edge_texts.append(f"{edge[0]}→{edge[1]}<br>Вес: {weight:.2f}<br>Загрузка: {congestion:.1f}%")

        edge_traces.append(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=style['width'], color=style['color']),
            hoverinfo='text',
            text=edge_texts,
            mode='lines',
            name=style['name'],
            showlegend=True
        ))

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
        text=list(G.nodes()),
        textposition="middle center",
        hoverinfo='text',
        textfont=dict(size=12, color='darkblue'),
        name='Вершины',
        showlegend=False
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    )

    return fig


with st.sidebar:
    st.title("Управление")

    st.subheader("Графы из data/")
    try:
        list_resp = requests.get(f"{API_URL}/graphs/list")
        if list_resp.status_code == 200:
            graphs = list_resp.json().get("graphs", [])
            if graphs:
                selected_graph = st.selectbox("Доступные графы", graphs, index=0)
                if st.button("Загрузить граф", type="primary", key="load_btn"):
                    load_resp = requests.post(f"{API_URL}/graphs/load", json={"filename": selected_graph})
                    if load_resp.status_code == 200:
                        st.success(f"Граф '{selected_graph}' загружен!")
                        st.session_state['graph_loaded'] = True
                        st.session_state['traffic'] = None
                        st.session_state['paths'] = None
                        st.rerun()
                    else:
                        st.error(f"Ошибка: {load_resp.json().get('detail')}")
            else:
                st.info("Папка data/ пуста. Создайте граф через 'Создать вручную' ниже.")
        else:
            st.warning("Не удалось получить список графов.")
    except Exception as e:
        st.error(f"Нет связи с бэкендом: {e}")

    st.divider()

    st.subheader("Время и условия")
    hour = st.slider("Час суток", 0, 23, 8)
    day_type = st.radio("Тип дня", ["Будни", "Выходные"], index=0)
    weather = st.selectbox("Погода", ["Ясно", "Дождь", "Снег"], index=0)

    day_type_code = 0 if day_type == "Будни" else 1
    weather_code = {"Ясно": 0, "Дождь": 1, "Снег": 2}[weather]

    st.divider()

    st.subheader("Маршрутизация")
    graph_info = None
    if st.session_state.get('graph_loaded', False):
        graph_info = get_graph_info()
        if not graph_info:
            import time

            time.sleep(0.5)
            graph_info = get_graph_info()

    if graph_info and graph_info.get('vertices'):
        nodes = graph_info['vertices']
        start_node = st.selectbox("Начало", nodes, index=0)
        end_node = st.selectbox("Конец", nodes, index=min(1, len(nodes) - 1))
        k_paths = st.slider("Количество путей (k)", 1, 5, 3)

        if st.button("Найти пути", type="primary"):
            st.session_state['paths'] = find_paths(start_node, end_node, k_paths)
            st.session_state['start'] = start_node
            st.session_state['end'] = end_node
    else:
        if not st.session_state.get('graph_loaded', False):
            st.warning("Граф не загружен. Выберите файл из списка выше.")
        else:
            st.info("Загрузка графа, подождите...")

    st.divider()

    if graph_info:
        st.subheader("Статистика")
        st.metric("Вершин", len(graph_info.get('vertices', [])))
        st.metric("Ребер", len(graph_info.get('edges', [])))

st.title("Маршрутизация с динамическими весами")
st.markdown("""
Система предсказания трафика на основе ML с визуализацией оптимальных маршрутов
""")

if st.button("Обновить предсказание трафика"):
    with st.spinner("Предсказание загруженности..."):
        traffic_data = predict_traffic(hour, day_type_code, weather_code)
        if traffic_data:
            st.session_state['traffic'] = traffic_data
            st.success("Трафик обновлен!")

time_str = f"{hour:02d}:00"
day_str = "Будни" if day_type_code == 0 else "Выходные"
weather_str = ["Ясно", "Дождь", "Снег"][weather_code]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Время", time_str)
with col2:
    st.metric("День", day_str)
with col3:
    st.metric("Погода", weather_str)
with col4:
    traffic_data = st.session_state.get('traffic')
    if traffic_data and traffic_data.get('predictions'):
        avg_congestion = np.mean([p['congestion_percent'] for p in traffic_data['predictions']])
        st.metric("Средняя загрузка", f"{avg_congestion:.1f}%")
    else:
        st.metric("Средняя загрузка", "—")

st.subheader("Визуализация сети")

viz_data = get_visualization_data()
if viz_data:
    paths = st.session_state.get('paths', None)
    traffic = st.session_state.get('traffic', None)

    fig = create_graph_visualization(viz_data, paths, traffic)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

paths_data = st.session_state.get('paths')
if paths_data and paths_data.get('paths'):
    st.subheader("Найденные маршруты")

    paths_data = st.session_state['paths']['paths']

    for path_info in paths_data:
        with st.expander(f"Путь #{path_info['index']} (Вес: {path_info['total_weight']:.2f})",
                         expanded=(path_info['index'] == 1)):
            col1, col2 = st.columns([2, 1])

            with col1:
                path_str = " → ".join(path_info['path'])
                st.write(f"**Маршрут:** {path_str}")

                st.write("**Ребра:**")
                for i in range(len(path_info['path']) - 1):
                    u, v = path_info['path'][i], path_info['path'][i + 1]
                    st.text(f"  {u} → {v}")

            with col2:
                st.metric("Общий вес", path_info['total_weight'])
                st.metric("Ребер", len(path_info['edges']))

traffic_data = st.session_state.get('traffic')
if traffic_data and traffic_data.get('predictions'):
    st.subheader("Загруженность ребер")

    traffic_df = pd.DataFrame(traffic_data['predictions'])
    traffic_df = traffic_df.sort_values('congestion_percent', ascending=False)


    def color_congestion(val):
        if val > 50:
            return 'background-color: #ff6b6b'
        elif val > 20:
            return 'background-color: #ffd93d'
        else:
            return 'background-color: #6bcb77'


    st.dataframe(
        traffic_df.style.applymap(color_congestion, subset=['congestion_percent']),
        use_container_width=True
    )

with st.expander("О системе"):
    st.markdown("""
    ### Визуализация:
    - 🔵 Синие ребра - низкая загрузка
    - 🟡 Желтые ребра - средняя загрузка  
    - 🔴 Красные ребра - высокая загрузка
    - 🔴 Жирные красные линии - выбранный путь
    """)