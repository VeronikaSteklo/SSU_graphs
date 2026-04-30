import streamlit as st
import requests
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import pandas as pd
import time

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


def get_floyd_warshall():
    try:
        response = requests.post(f"{API_URL}/paths/floyd-warshall")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Ошибка алгоритма Флойда-Уоршелла: {e}")
    return None


def get_algorithm_steps(algo_name, start_node, end_node):
    """Запрос пошаговых логов алгоритма с бэкенда"""
    try:
        response = requests.post(
            f"{API_URL}/paths/visualize-steps/{algo_name}",
            json={"start": start_node, "end": end_node, "k": 1}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка: {response.json().get('detail', 'Неизвестная ошибка')}")
    except Exception as e:
        st.error(f"Ошибка получения шагов: {e}")
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


def create_graph_visualization(viz_data, paths=None, congestion_data=None, highlight_nodes=None, highlight_edges=None):
    """Создание интерактивной визуализации графа с группировкой рёбер по цвету и подсветкой шагов"""
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
                if not viz_data.get('is_directed', False):
                    path_edges.add((path[i + 1], path[i]))

    hl_edges_set = set()
    if highlight_edges:
        for edge in highlight_edges:
            if len(edge) >= 2:
                u, v = edge[0], edge[1]
                hl_edges_set.add((u, v))
                if not viz_data.get('is_directed', False):
                    hl_edges_set.add((v, u))

    edge_groups = {'active': [], 'path': [], 'high': [], 'medium': [], 'low': []}

    for edge in G.edges():
        congestion = edge_congestion.get(edge, 0)

        if edge in hl_edges_set:
            edge_groups['active'].append(edge)
        elif edge in path_edges:
            edge_groups['path'].append(edge)
        elif congestion > 50:
            edge_groups['high'].append(edge)
        elif congestion > 20:
            edge_groups['medium'].append(edge)
        else:
            edge_groups['low'].append(edge)

    styles = {
        'active': {'color': 'rgb(255, 140, 0)', 'width': 5, 'name': 'Текущий шаг алгоритма'},
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

    hl_nodes_set = set(highlight_nodes) if highlight_nodes else set()

    node_x, node_y, normal_texts = [], [], []
    hl_node_x, hl_node_y, hl_texts = [], [], []

    for node in G.nodes():
        x, y = pos[node]
        if node in hl_nodes_set:
            hl_node_x.append(x)
            hl_node_y.append(y)
            hl_texts.append(node)
        else:
            node_x.append(x)
            node_y.append(y)
            normal_texts.append(node)

    node_traces = []

    if node_x:
        node_traces.append(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
            text=normal_texts,
            textposition="middle center",
            hoverinfo='text',
            textfont=dict(size=12, color='darkblue'),
            name='Вершины',
            showlegend=False
        ))

    if hl_node_x:
        node_traces.append(go.Scatter(
            x=hl_node_x, y=hl_node_y,
            mode='markers+text',
            marker=dict(size=26, color='orange', line=dict(width=3, color='red')),
            text=hl_texts,
            textposition="middle center",
            hoverinfo='text',
            textfont=dict(size=13, color='white', weight='bold'),
            name='Активные вершины',
            showlegend=True
        ))

    fig = go.Figure(
        data=edge_traces + node_traces,
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
                        st.session_state['floyd_matrix'] = None
                        st.session_state['algo_steps'] = None
                        st.rerun()
                    else:
                        st.error(f"Ошибка: {load_resp.json().get('detail')}")
            else:
                st.info("Папка data/ пуста.")
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
            st.session_state['algo_steps'] = None
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

    st.divider()
    st.subheader("Запуск алгоритмов")
    algo_choice = st.selectbox(
        "Алгоритм",
        ["dijkstra", "kruskal", "bellman_ford", "max_flow", "floyd_warshall"],
        format_func=lambda x: {
            "dijkstra": "Дейкстра (Кратчайший путь)",
            "kruskal": "Краскал (MST)",
            "bellman_ford": "Беллман-Форд",
            "max_flow": "Эдмондс-Карп",
            "floyd_warshall": "Флойд-Уоршелл"
        }.get(x, x)
    )

    if graph_info and graph_info.get('vertices'):
        col_start, col_end = st.columns(2)
        with col_start:
            a_start = st.selectbox("Откуда", graph_info['vertices'], key="algo_start")
        with col_end:
            a_end = st.selectbox("Куда", graph_info['vertices'], index=min(1, len(graph_info['vertices']) - 1),
                                 key="algo_end")

        if st.button("Запустить визуализацию", type="primary"):
            with st.spinner("Запрашиваем историю шагов..."):
                steps_res = get_algorithm_steps(algo_choice, a_start, a_end)
                if steps_res and "steps" in steps_res:
                    st.session_state['algo_steps'] = steps_res['steps']

                    st.session_state['algo_result'] = steps_res.get('result')

                    st.session_state['run_animation'] = True
                    st.rerun()

st.title("Маршрутизация с динамическими весами")
st.markdown("Система предсказания трафика на основе ML с визуализацией оптимальных маршрутов и алгоритмов")

if st.button("Обновить предсказание трафика"):
    with st.spinner("Предсказание загруженности..."):
        traffic_data = predict_traffic(hour, day_type_code, weather_code)
        if traffic_data:
            st.session_state['traffic'] = traffic_data

            st.session_state['algo_steps'] = None
            st.session_state['algo_result'] = None

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

viz_data = get_visualization_data()
if viz_data:
    algo_steps = st.session_state.get('algo_steps')
    algo_result = st.session_state.get('algo_result')

    if algo_steps:
        st.subheader("Визуализация алгоритма")
        total_steps = len(algo_steps)

        if st.session_state.get('run_animation', False):
            if st.button("⏭️ Пропустить анимацию (сразу к результату)", key="skip_anim_btn"):
                st.session_state['run_animation'] = False
                st.rerun()

        msg_placeholder = st.empty()
        graph_placeholder = st.empty()

        if st.session_state.get('run_animation', False):
            for step_idx, current_step in enumerate(algo_steps):
                if not st.session_state.get('run_animation', False):
                    break

                msg_placeholder.info(f"**Шаг {step_idx + 1} из {total_steps}:** {current_step.get('message', '')}")

                hl_nodes = current_step.get('nodes', [])
                hl_edges = current_step.get('edges', [])

                fig = create_graph_visualization(viz_data, highlight_nodes=hl_nodes, highlight_edges=hl_edges)
                if fig:
                    graph_placeholder.plotly_chart(fig, width='stretch', key=f"anim_step_{step_idx}")

                time.sleep(1.5)

            st.session_state['run_animation'] = False
            st.rerun()

        if not st.session_state.get('run_animation', False):
            last_step = algo_steps[-1]
            msg_placeholder.success(f"**Завершено.** {last_step.get('message', '')}")

            fig_final = create_graph_visualization(
                viz_data,
                highlight_nodes=last_step.get('nodes', []),
                highlight_edges=last_step.get('edges', [])
            )
            if fig_final:
                graph_placeholder.plotly_chart(fig_final, width='stretch', key="final_step_graph")

            if algo_result:
                st.divider()
                st.subheader("Итоговые результаты алгоритма")

                if "distance" in algo_result and "path" in algo_result:
                    st.write(f"**Кратчайшее расстояние:** `{algo_result['distance']}`")
                    path_str = " → ".join(algo_result['path']) if algo_result['path'] else "Путь не найден"
                    st.write(f"**Оптимальный путь:** {path_str}")

                elif "total_weight" in algo_result and "mst_edges" in algo_result:
                    st.write(f"**Общий вес минимального остовного дерева (MST):** `{algo_result['total_weight']}`")
                    with st.expander("Показать рёбра, вошедшие в дерево", expanded=True):
                        for edge in algo_result['mst_edges']:
                            st.write(f"- Узел `{edge[0]}` ↔ Узел `{edge[1]}`")

                elif "distances" in algo_result:
                    st.write("**Кратчайшие расстояния от стартовой вершины:**")
                    dist_df = pd.DataFrame(list(algo_result['distances'].items()), columns=["Вершина", "Расстояние"])
                    st.dataframe(dist_df, use_container_width=True)

                elif "message" in algo_result:
                    st.info(algo_result["message"])

    else:
        st.subheader("Визуализация сети")
        paths = st.session_state.get('paths', None)
        traffic = st.session_state.get('traffic', None)

        fig = create_graph_visualization(viz_data, paths, traffic)
        if fig:
            st.plotly_chart(fig, width='stretch')

paths_data = st.session_state.get('paths')
if paths_data and paths_data.get('paths') and not st.session_state.get('algo_steps'):
    st.divider()
    st.subheader("Найденные маршруты")

    paths_list = st.session_state['paths']['paths']

    for path_info in paths_list:
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
if traffic_data and traffic_data.get('predictions') and not st.session_state.get('algo_steps'):
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

st.divider()
st.subheader("Матрица кратчайших путей (Floyd-Warshall)")

if st.button("Рассчитать все пути"):
    with st.spinner("Вычисление матрицы..."):
        floyd_data = get_floyd_warshall()
        if floyd_data:
            st.session_state['floyd_matrix'] = floyd_data
            st.success("Матрица успешно рассчитана!")

if st.session_state.get('floyd_matrix'):
    data = st.session_state['floyd_matrix']
    nodes = data['nodes']
    matrix = data['matrix']

    df_matrix = pd.DataFrame.from_dict(matrix, orient='index')
    df_matrix = df_matrix[nodes]

    st.write("Минимальные веса между вершинами:")


    def highlight_inf(val):
        color = 'color: #d32f2f' if val == 'inf' else 'color: #1e88e5'
        return color


    st.dataframe(
        df_matrix.style.applymap(highlight_inf),
        use_container_width=True
    )

    if st.checkbox("Показать пояснение к матрице"):
        st.info(
            "Строки — это точки отправления, столбцы — точки прибытия. Значение 'inf' означает, что пути между узлами не существует.")
