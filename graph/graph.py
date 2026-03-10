import copy
import json
import random
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


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
            if find(u) != find(v):
                union(u, v)
                mst_graph.add_edge(u, v, weight)
                total_weight += weight
                edges_count += 1

        return mst_graph, total_weight