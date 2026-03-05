from graph import Graph, GraphError
import sys


class GraphInterface:
    def __init__(self):
        self.graph = None

        self.menu_creation = {
            "1": self.create_empty,
            "2": self.load_json,
            "3": self.generate_random,
            "4": self.combine_two_jsons,
            "5": self.combine_with_another,
            "0": self.back
        }

        self.menu_cats = {
            "1": self.submenu_creation,
            "2": self.submenu_edit,
            "3": self.submenu_view,
            "4": self.submenu_files,
            "0": self.exit_app
        }

        self.menu_edit = {
            "1": self.add_node,
            "2": self.add_link,
            "3": self.delete_node,
            "4": self.delete_link,
            "5": self.change_weight,
            "0": self.back
        }

        self.menu_view = {
            "1": self.show_adj,
            "2": self.draw,
            "3": self.show_degrees,
            "4": self.show_non_adjacent,
            "5": self.show_is_tree_forest,
            "6": self.show_shortest_to_set,
            "0": self.back
        }

        self.menu_files = {
            "1": self.save_json,
            "2": self.load_json,
            "0": self.back
        }

    def run(self):
        print("--- Система управления графами ---")
        while True:
            try:
                if self.graph is None:
                    self.submenu_creation()
                else:
                    self.menu_main_categories()
            except KeyboardInterrupt:
                self.exit_app()
            except Exception as e:
                print(f"\n[ОШИБКА]: {e}")

    def menu_main_categories(self):
        print("\n" + "=" * 30)
        print(f" ТЕКУЩИЙ ГРАФ: {repr(self.graph)}")
        print("=" * 30)
        print("1. [Создание] (Начать заново / Сгенерировать другой)")
        print("2. [Редактирование] (Вершины / Ребра)")
        print("3. [Просмотр] (Список смежности / Рисунок)")
        print("4. [Файлы] (Сохранить результат)")
        print("0. Выход")

        ch = input("\nВыберите категорию: ")
        if ch in self.menu_cats:
            self.menu_cats[ch]()
        else:
            print("Неверная категория.")

    def _execute_from_menu(self, menu, choice):
        """Вспомогательный метод для вызова функций из словарей."""
        if choice == "0":
            return False
        if choice in menu:
            menu[choice]()
            return True
        print("Неверный выбор.")
        return True

    def submenu_creation(self):
        print("\n--- СОЗДАНИЕ ---")
        print("1. Создать пустой")
        print("2. Создать из json")
        print("3. Сгенерировать новый случайный")
        print("4. Объединить 2 графа из json")
        print("5. Объединить текущий с другим (из json)")
        print("0. " + ("Назад" if self.graph else "Выход"))
        ch = input("> ")
        if ch == "0" and not self.graph: self.exit_app()
        self._execute_from_menu(self.menu_creation, ch)

    def submenu_edit(self):
        print("\n--- РЕДАКТИРОВАНИЕ ---")
        print("1. Добавить вершину")
        print("2. Добавить ребро")
        print("3. Удалить вершину")
        print("4. Удалить ребро")
        if self.graph.is_weighted:
            print("5. Изменить вес")
        print("0. Назад")
        ch = input("> ")
        self._execute_from_menu(self.menu_edit, ch)

    def submenu_view(self):
        print("\n--- ПРОСМОТР ---")
        print("1. Список смежности")
        print("2. Визуализация")
        print("3. Степени всех вершин")
        print("4. Найти не смежные вершины")
        print("5. Проверка на дерево/лес")
        print("6. Расстояние до множества вершин")
        print("0. Назад")
        ch = input("> ")
        self._execute_from_menu(self.menu_view, ch)

    def submenu_files(self):
        print("\n--- ФАЙЛЫ ---")
        print("1. Сохранить в JSON")
        print("2. Загрузить другой файл")
        print("0. Назад")
        ch = input("> ")
        self._execute_from_menu(self.menu_files, ch)

    def back(self):
        pass

    def _ensure_graph(self):
        if self.graph is None:
            raise GraphError("Сначала создайте граф.")

    def create_empty(self):
        d = input("Ориентированный? (y/n): ").lower() == 'y'
        w = input("Взвешенный? (y/n): ").lower() == 'y'
        self.graph = Graph(is_directed=d, is_weighted=w)
        print("Граф создан.")

    def load_json(self):
        fname = input("Имя файла: ")
        self.graph = Graph.from_json(fname)
        print("Загружено.")

    def generate_random(self):
        n = int(input("Вершин: "))
        m = int(input("Ребер: "))
        d = input("Ориентированный? (y/n): ").lower() == 'y'
        w = input("Взвешенный? (y/n): ").lower() == 'y'
        self.graph = Graph.from_random(n, m, d, w)
        print("Сгенерировано.")

    def combine_with_another(self, ):
        """Обработчик построения объединения."""
        self._ensure_graph()
        print("\nДля объединения нужно загрузить второй граф из файла.")
        fname = input("Введите имя JSON-файла второго графа: ")

        try:
            g2 = Graph.from_json(fname)
            self.graph = Graph.union(self.graph, g2)
            print(f"Объединение выполнено успешно. Текущий граф теперь содержит {len(self.graph._adj_list)} вершин.")
        except FileNotFoundError:
            print("[ОШИБКА]: Файл не найден.")
        except Exception as e:
            print(f"[ОШИБКА при объединении]: {e}")

    def combine_two_jsons(self):
        file1 = input("Введите имя первого JSON-файла: ")
        file2 = input("Введите имя второго JSON-файла: ")

        try:
            g1 = Graph.from_json(file1)
            g2 = Graph.from_json(file2)

            self.graph = Graph.union(g1, g2)

            print(f"\n[УСПЕХ]: Графы из '{file1}' и '{file2}' объединены.")
            print(f"Итого вершин: {len(self.graph._adj_list)}")

        except FileNotFoundError as e:
            print(f"[ОШИБКА]: Один из файлов не найден ({e})")
        except Exception as e:
            print(f"[ОШИБКА]: {e}")

    def add_node(self):
        self.graph.add_vertex(input("Имя вершины: "))

    def add_link(self):
        u, v = input("Откуда: "), input("Куда: ")
        w = float(input("Вес: ")) if self.graph.is_weighted else 1.0
        self.graph.add_edge(u, v, w)

    def delete_node(self):
        self.graph.remove_vertex(input("Имя вершины: "))

    def delete_link(self):
        u, v = input("Узел 1: "), input("Узел 2: ")
        self.graph.remove_edge(u, v)

    def change_weight(self):
        u, v, weight = input("Узел 1: "), input("Узел 2: "), float(input("Новый вес: "))
        while weight is None or weight < 0:
            raise GraphError(f"Введите корректный вес!")
        weight = float(input("Новый вес: "))
        self.graph.change_weight(u, v, weight)

    def show_adj(self):
        print(self.graph)

    def draw(self):
        self.graph.visualize()

    def save_json(self):
        fname = input("Имя файла: ")
        self.graph.to_json(fname)
        print("Сохранено.")

    def show_degrees(self):
        self._ensure_graph()
        degrees = self.graph.get_vertex_degrees()

        print("\n--- СТЕПЕНИ ВЕРШИН ---")
        if not degrees:
            print("Граф пуст.")
            return

        for node, val in degrees.items():
            if isinstance(val, dict):
                print(f"Вершина {node: <5} | Входящая: {val['in']}, Исходящая: {val['out']}, Всего: {val['total']}")
            else:
                print(f"Вершина {node: <5} | Степень: {val}")

    def show_non_adjacent(self):
        self._ensure_graph()
        v = input("Введите имя целевой вершины: ")
        try:
            non_adj = self.graph.get_non_adjacent_vertices(v)
            if not non_adj:
                print(f"Вершина '{v}' смежна со всеми остальными вершинами.")
            else:
                print(f"Вершины, не смежные с '{v}': {', '.join(non_adj)}")
        except GraphError as e:
            print(f"[ОШИБКА]: {e}")

    def show_is_tree_forest(self):
        tree = self.graph.is_tree_or_forest()
        print(f"Граф — {tree}")

    def show_shortest_to_set(self):
        self._ensure_graph()
        raw_input = input("Введите имена целевых вершин через пробел: ")
        target_list = raw_input.split()

        if not target_list:
            raise GraphError("Множество целей не может быть пустым.")

        try:
            results = self.graph.find_shortest_to_set_universal(target_list)

            print(f"\nКратчайшие расстояния до ближайшей из {target_list}:")
            for node, dist in results.items():
                print(f"Вершина {node: <10} | Расстояние: {dist}")
        except Exception as e:
            print(f"[ОШИБКА]: {e}")

    def exit_app(self):
        print("\nЗавершение работы.")
        sys.exit(0)