import copy
import heapq
import json
import random
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

import numpy as np


class GraphError(Exception):
    """Базовый класс для ошибок в работе с графом."""
    pass


class Graph:
    def __init__(self, is_directed: bool = False, is_weighted: bool = False):
        """1. Конструктор по умолчанию."""
        self.is_directed = is_directed
        self.is_weighted = is_weighted
        self._adj_list = {}

    @classmethod
    def from_json(cls, filename: str):
        """Загрузка графа из JSON файла."""
        path = "data/" + filename + ".json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл {filename} не найден.")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instance = cls(is_directed=data['is_directed'], is_weighted=data['is_weighted'])
        instance._adj_list = data['adj_list']
        return instance

    @classmethod
    def from_copy(cls, other):
        """3. Конструктор-копия."""
        if not isinstance(other, Graph):
            raise TypeError("Копируемый объект должен быть экземпляром класса Graph.")

        instance = cls(other.is_directed, other.is_weighted)
        instance._adj_list = copy.deepcopy(other._adj_list)
        return instance

    @classmethod
    def from_random(cls, num_vertices: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False):
        """4. Специфический конструктор для генерации случайного графа."""
        if num_vertices <= 0:
            raise ValueError("Количество вершин должно быть положительным.")

        max_possible_edges = num_vertices * num_vertices if is_directed else (num_vertices * (num_vertices + 1)) // 2
        if num_edges > max_possible_edges:
            raise GraphError(f"Слишком много ребер ({num_edges}) для {num_vertices} вершин.")

        instance = cls(is_directed, is_weighted)
        vertices = [f"V{i}" for i in range(1, num_vertices + 1)]

        for v in vertices:
            instance.add_vertex(v)

        added_edges = 0
        while added_edges < num_edges:
            u = random.choice(vertices)
            if is_directed:
                v = random.choice(vertices)
            else:
                other_vertices = [v for v in vertices if v != u]
                v = random.choice(other_vertices)

            if v not in instance._adj_list[u]:
                weight = round(random.uniform(1.0, 10.0), 1) if is_weighted else 1.0
                instance.add_edge(u, v, weight)
                added_edges += 1

        return instance

    @classmethod
    def union(cls, g1, g2):
        """ Построение объединения двух графов. """
        if g1.is_directed != g2.is_directed:
            raise GraphError("Нельзя объединить ориентированный и неориентированный графы.")

        new_is_weighted = g1.is_weighted or g2.is_weighted
        new_graph = cls(is_directed=g1.is_directed, is_weighted=new_is_weighted)

        for v in g1._adj_list:
            new_graph.add_vertex(v)
        for u, v, w in g1.get_edge_list():
            new_graph.add_edge(u, v, w)

        for v in g2._adj_list:
            if v not in new_graph._adj_list:
                new_graph.add_vertex(v)

        for u, v, w in g2.get_edge_list():
            if v in new_graph._adj_list[u]:
                current_weight = new_graph._adj_list[u][v]
                new_graph._adj_list[u][v] = current_weight + w
                if not new_graph.is_directed:
                    new_graph._adj_list[v][u] = current_weight + w
            else:
                new_graph.add_edge(u, v, w)

        return new_graph

    def get_edge_list(self):
        """Возвращает список ребер в формате (откуда, куда, вес)."""
        edges = []
        visited = set()
        for u in self._adj_list:
            for v, weight in self._adj_list[u].items():
                edge_id = (u, v) if self.is_directed else tuple(sorted((u, v)))
                if self.is_directed or edge_id not in visited:
                    edges.append((u, v, weight))
                    visited.add(edge_id)
        return edges

    def get_edge_list_for_api(self):
        raw_edges = self.get_edge_list()
        return [
            {"source": str(u), "target": str(v), "weight": float(w)}
            for u, v, w in raw_edges
        ]

    def add_vertex(self, v: str):
        """Добавляет вершину в граф."""
        if v in self._adj_list:
            raise GraphError(f"Вершина '{v}' уже существует.")
        self._adj_list[v] = {}

    def add_edge(self, u: str, v: str, weight: float = 1.0):
        """Добавляет ребро (дугу) между u и v."""
        if u not in self._adj_list:
            raise GraphError(f"Вершины '{u}' не существует.")
        if v not in self._adj_list:
            raise GraphError(f"Вершины '{v}' не существует.")

        if not self.is_directed and u == v:
            raise GraphError(f"В неориентированном графе не должно быть петель")

        if v in self._adj_list[u]:
            raise GraphError(f"Ребро из '{u}' в '{v}' уже существует.")

        self._adj_list[u][v] = weight
        if not self.is_directed:
            if u != v:
                self._adj_list[v][u] = weight

    def change_weight(self, u: str, v: str, weight: float = 1.0):
        if not self.is_weighted:
            raise GraphError(f"Граф невзвешенный, весов нет!")
        if u not in self._adj_list:
            raise GraphError(f"Вершины '{u}' не существует.")
        if v not in self._adj_list:
            raise GraphError(f"Вершины '{v}' не существует.")

        self._adj_list[u][v] = weight

    def remove_edge(self, u: str, v: str):
        """Удаляет ребро (дугу) из u в v."""
        if u not in self._adj_list:
            raise GraphError(f"Вершина '{u}' не найдена.")
        if v not in self._adj_list:
            raise GraphError(f"Вершина '{v}' не найдена.")
        if v not in self._adj_list[u]:
            raise GraphError(f"Ребро из '{u}' в '{v}' не найдено.")

        del self._adj_list[u][v]

        if not self.is_directed:
            if v in self._adj_list and u in self._adj_list[v]:
                del self._adj_list[v][u]

    def remove_vertex(self, v: str):
        """Удаляет вершину и все инцидентные ей ребра."""
        if v not in self._adj_list:
            raise GraphError(f"Вершина '{v}' не найдена.")

        del self._adj_list[v]

        for source, neighbors in self._adj_list.items():
            if v in neighbors:
                del neighbors[v]

    def to_json(self, filename: str):
        """Сохранение графа в JSON файл."""

        data = {
            "is_directed": self.is_directed,
            "is_weighted": self.is_weighted,
            "adj_list": self._adj_list
        }

        with open("data/" + filename + ".json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def __str__(self) -> str:
        """Пользовательское представление графа (список смежности)."""
        if not self._adj_list:
            return "Граф пуст."

        lines = []
        for u, neighbors in self._adj_list.items():
            if not neighbors:
                lines.append(f"{u}: -")
            else:
                neighbor_parts = []
                for v, w in neighbors.items():
                    part = f"{v}({w})" if self.is_weighted else str(v)
                    neighbor_parts.append(part)

                lines.append(f"{u}: {', '.join(neighbor_parts)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Техническое представление графа."""
        return f"Graph({"ориентированный" if self.is_directed else "неориентированный"}, {"взвешенный" if self.is_weighted else "невзвешенный"}, количество вершин={len(self._adj_list)})"

    def visualize(self):
        """ Визуализирует граф """

        nx_graph = nx.DiGraph() if self.is_directed else nx.Graph()
        nx_graph.add_nodes_from(self._adj_list.keys())
        for u, neighbors in self._adj_list.items():
            for v, weight in neighbors.items():
                nx_graph.add_edge(u, v, weight=weight)

        pos = nx.spring_layout(nx_graph, k=2.0, iterations=50)

        plt.figure(figsize=(10, 7))

        nx.draw_networkx_nodes(nx_graph, pos, node_size=1300, node_color='lightblue')

        nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_family='sans-serif', font_weight='bold')

        if self.is_directed:
            nx.draw_networkx_edges(
                nx_graph, pos,
                edgelist=nx_graph.edges(),
                arrowstyle='->',
                arrowsize=30,
                edge_color='gray',
                width=2,
                node_size=1500,
                connectionstyle='arc3, rad=0.1'
            )
        else:
            nx.draw_networkx_edges(nx_graph, pos, width=2, edge_color='gray')

        if self.is_weighted:
            edge_labels = nx.get_edge_attributes(nx_graph, 'weight')

            if self.is_directed:
                nx.draw_networkx_edge_labels(
                    nx_graph, pos,
                    edge_labels=edge_labels,
                    label_pos=0.3,
                    font_size=9,
                    rotate=True,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2')
                )
            else:
                nx.draw_networkx_edge_labels(
                    nx_graph, pos,
                    edge_labels=edge_labels,
                    font_size=9
                )

        plt.title(f"Визуализация: {'Ориентированный' if self.is_directed else 'Неориентированный'} граф")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def get_vertex_degrees(self):
        """ Возвращает словарь со степенями всех вершин.
        Для ориентированного графа возвращает (входящая, исходящая, общая). """
        degrees = {}
        for v in self._adj_list:
            if not self.is_directed:
                degrees[v] = len(self._adj_list[v])
            else:
                out_degree = len(self._adj_list[v])
                in_degree = sum(1 for u in self._adj_list if v in self._adj_list[u])
                degrees[v] = {
                    "in": in_degree,
                    "out": out_degree,
                    "total": in_degree + out_degree
                }
        return degrees

    def get_non_adjacent_vertices(self, v: str):
        """ Возвращает список всех вершин, не смежных с данной вершиной v. """
        if v not in self._adj_list:
            raise GraphError(f"Вершина '{v}' не найдена в графе.")

        adjacent = set(self._adj_list[v].keys())

        non_adjacent = [
            node for node in self._adj_list
            if node != v and node not in adjacent
        ]
        return non_adjacent

    def is_tree_or_forest(self):
        """ Проверяет, является ли орграф деревом, лесом или ни тем, ни другим. """
        if not self.is_directed:
            return "Метод предназначен только для ориентированных графов."

        vertices = list(self._adj_list.keys())
        if not vertices:
            return "Пустой граф можно считать лесом."

        degrees = self.get_vertex_degrees()
        for v in vertices:
            if degrees[v]['in'] > 1:
                return "Не является ни деревом, ни лесом (у вершины степень захода > 1)."

        visited = set()
        rec_stack = set()

        def has_cycle(v):
            visited.add(v)
            rec_stack.add(v)
            for neighbor in self._adj_list[v]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(v)
            return False

        for node in vertices:
            if node not in visited:
                if has_cycle(node):
                    return "Не является ни деревом, ни лесом (содержит цикл)."

        roots = [v for v in vertices if degrees[v]['in'] == 0]

        if len(roots) == 1:
            return "дерево."
        elif len(roots) > 1:
            return "лес (несколько корней)."
        else:
            return "не является ни деревом, ни лесом."

    def find_shortest_to_set_universal(self, target_set: list) -> dict:
        """ Находит кратчайшее расстояние от каждой вершины до ближайшей из target_set. """

        for v in target_set:
            if v not in self._adj_list:
                raise GraphError(f"Вершина '{v}' не найдена.")

        distances = {v: float('inf') for v in self._adj_list}
        queue = deque()

        for v in target_set:
            distances[v] = 0
            queue.append(v)

        if self.is_directed:
            predecessors = {v: [] for v in self._adj_list}
            for u in self._adj_list:
                for v in self._adj_list[u]:
                    predecessors[v].append(u)

        while queue:
            current_v = queue.popleft()

            neighbors = predecessors[current_v] if self.is_directed else self._adj_list[current_v]

            for neighbor in neighbors:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_v] + 1
                    queue.append(neighbor)

        return distances

    def find_mst_kruskal(self):
        """ Находит минимальное остовное дерево (каркас) методом Краскала. """
        if self.is_directed:
            raise GraphError("Алгоритм Краскала предназначен для неориентированных графов.")
        if not self.is_weighted:
            raise GraphError("Для поиска MST граф должен быть взвешенным.")

        num_vertices = len(self._adj_list)
        if num_vertices == 0:
            return Graph(False, True), 0

        parent = {v: v for v in self._adj_list}
        rank = {v: 0 for v in self._adj_list}

        def find(v):
            if parent[v] == v:
                return v
            parent[v] = find(parent[v])
            return parent[v]

        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                if rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                elif rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                else:
                    parent[root_v] = root_u
                    rank[root_u] += 1
                return True
            return False

        edges = self.get_edge_list()
        edges.sort(key=lambda x: x[2])

        mst_graph = Graph(is_directed=False, is_weighted=True)
        for v in self._adj_list:
            mst_graph.add_vertex(v)

        edges_count = 0
        total_weight = 0

        for u, v, weight in edges:
            if edges_count == num_vertices - 1:
                break
            if find(u) != find(v):
                union(u, v)
                mst_graph.add_edge(u, v, weight)
                total_weight += weight
                edges_count += 1

        return mst_graph, total_weight

    def find_k_shortest_paths(self, start: str, end: str, k: int):
        """ Поиск k кратчайших путей между u и v (Алгоритм Йена + Дейкстра). """
        if start not in self._adj_list or end not in self._adj_list:
            raise GraphError("Вершины не найдены")

        def dijkstra(temp_graph, start, end):
            distances = {node: float('inf') for node in temp_graph._adj_list}
            distances[start] = 0
            paths = {node: [] for node in temp_graph._adj_list}
            paths[start] = [start]
            pq = [(0, start)]

            visited = set()
            while pq:
                (dist, current) = heapq.heappop(pq)
                if current in visited:
                    continue
                if current == end:
                    return (dist, paths[end])
                visited.add(current)

                for neighbor, weight in temp_graph._adj_list[current].items():
                    if neighbor in visited:
                        continue

                    new_dist = dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        paths[neighbor] = paths[current] + [neighbor]
                        heapq.heappush(pq, (new_dist, neighbor))
            return None

        A = []
        B = []

        initial_path = dijkstra(self, start, end)
        if not initial_path: return []
        A.append(initial_path)

        for i in range(1, k):
            for j in range(len(A[-1][1]) - 1):
                spur_node = A[-1][1][j]
                root_path = A[-1][1][:j + 1]

                edges_removed = []
                for path_data in A:
                    path = path_data[1]
                    if len(path) > j and root_path == path[:j + 1]:
                        u, v = path[j], path[j + 1]
                        if v in self._adj_list[u]:
                            weight = self._adj_list[u][v]
                            edges_removed.append((u, v, weight))
                            self.remove_edge(u, v)

                spur_path_data = dijkstra(self, spur_node, end)
                if spur_path_data:
                    total_path = root_path[:-1] + spur_path_data[1]
                    total_dist = sum(self.get_edge_weight(total_path[m], total_path[m + 1])
                                     for m in range(len(total_path) - 1))
                    if (total_dist, total_path) not in B:
                        heapq.heappush(B, (total_dist, total_path))

                for u, v, w in edges_removed:
                    self.add_edge(u, v, w)

            if not B: break
            A.append(heapq.heappop(B))

        return A

    def get_edge_weight(self, u, v):
        return self._adj_list[u][v]

    def all_pairs_shortest_paths_floyd(self):
        """ Поиск кратчайших путей для всех пар вершин (Алгоритм Флойда-Уоршелла). """
        nodes = list(self._adj_list.keys())
        dist = {u: {v: float('inf') for v in nodes} for u in nodes}

        for u in nodes:
            dist[u][u] = 0
            for v, weight in self._adj_list[u].items():
                dist[u][v] = weight

        for k in nodes:  # k нельзя вставлять в конец
            for i in nodes:
                for j in nodes:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def find_negative_cycle_pairs_bellman(self):
        """ Поиск всех пар вершин, между которыми существует путь сколь угодно малой длины.
         (Алгоритм Беллмана-Форда для поиска отрицательных циклов). """
        nodes = list(self._adj_list.keys())
        inf_paths = []

        for start_node in nodes:
            dist = {node: float('inf') for node in nodes}
            dist[start_node] = 0

            for _ in range(len(nodes) - 1):
                for u, v, w in self.get_edge_list():
                    if dist[u] != float('inf') and dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w

            affected_by_cycle = set()
            for _ in range(len(nodes)):
                for u, v, w in self.get_edge_list():
                    if dist[u] != float('inf') and dist[u] + w < dist[v]:
                        dist[v] = float('-inf')
                        affected_by_cycle.add(v)

            for target_node in nodes:
                if dist[target_node] == float('-inf'):
                    inf_paths.append((start_node, target_node))

        return list(set(inf_paths))

    def find_max_flow(self, source: str, sink: str):
        """Находит максимальный поток из истока в сток (Алгоритм Эдмондса-Карпа)."""
        if source not in self._adj_list or sink not in self._adj_list:
            raise GraphError("Указанные вершины истока или стока не найдены.")

        residual_adj = {u: {} for u in self._adj_list}
        for u in self._adj_list:
            for v, weight in self._adj_list[u].items():
                residual_adj[u][v] = weight
                if u not in residual_adj[v]:
                    residual_adj[v][u] = 0

        max_flow = 0
        parent = {}

        def bfs():
            visited = {node: False for node in residual_adj}
            queue = deque([source])
            visited[source] = True
            while queue:
                u = queue.popleft()
                for v, capacity in residual_adj[u].items():
                    if not visited[v] and capacity > 0:
                        parent[v] = u
                        visited[v] = True
                        if v == sink:
                            return True
                        queue.append(v)
            return False

        while bfs():
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, residual_adj[parent[s]][s])
                s = parent[s]

            max_flow += path_flow
            v = sink
            while v != source:
                u = parent[v]
                residual_adj[u][v] -= path_flow
                residual_adj[v][u] += path_flow
                v = parent[v]

        return max_flow

    def get_weighted_random_walks(self, walk_length: int, num_walks: int):
        """Блуждания с учетом весов ребер."""
        walks = []
        nodes = list(self._adj_list.keys())

        for _ in range(num_walks):
            random.shuffle(nodes)
            for start_node in nodes:
                walk = [start_node]
                while len(walk) < walk_length:
                    cur = walk[-1]
                    neighbors_dict = self._adj_list[cur]

                    if not neighbors_dict:
                        break

                    # Извлекаем соседей и их веса
                    neighbors = list(neighbors_dict.keys())
                    weights = list(neighbors_dict.values())

                    # Нормализуем веса, чтобы получить вероятности
                    total_weight = sum(weights)
                    probabilities = [w / total_weight for w in weights]

                    # Выбираем следующую вершину согласно распределению
                    next_node = np.random.choice(neighbors, p=probabilities)
                    walk.append(next_node)
                walks.append(walk)
        return walks


def generate_social_graph(num_communities=3, nodes_per_comm=15):
    """ Генерирует граф с четко выраженными сообществами. """
    g = Graph(is_directed=False, is_weighted=True)
    communities = []

    node_id = 0
    for c in range(num_communities):
        comm_nodes = []
        for _ in range(nodes_per_comm):
            name = f"User_{node_id}"
            g.add_vertex(name)
            comm_nodes.append(name)
            node_id += 1
        communities.append(comm_nodes)

    all_nodes = [n for comm in communities for n in comm]
    for i, u in enumerate(all_nodes):
        for j, v in enumerate(all_nodes):
            if i >= j: continue
            same = any(u in c and v in c for c in communities)

            if same and random.random() < 0.4:
                weight = random.uniform(10.0, 20.0)
                g.add_edge(u, v, weight=weight)
            elif not same and random.random() < 0.02:
                weight = random.uniform(0.5, 1.0)
                g.add_edge(u, v, weight=weight)
    return g