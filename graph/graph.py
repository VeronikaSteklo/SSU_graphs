import copy
import json
import random
import os
import networkx as nx
import matplotlib.pyplot as plt


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
