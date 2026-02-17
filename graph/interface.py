from graph import Graph, GraphError
import sys


class GraphInterface:
    def __init__(self):
        self.graph = None

        self.menu_creation = {
            "1": self.create_empty,
            "2": self.load_json,
            "3": self.generate_random,
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
            "0": self.back
        }

        self.menu_view = {
            "1": self.show_adj,
            "2": self.draw,
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
        print("0. Назад")
        ch = input("> ")
        self._execute_from_menu(self.menu_edit, ch)

    def submenu_view(self):
        print("\n--- ПРОСМОТР ---")
        print("1. Список смежности")
        print("2. Визуализация")
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

    def show_adj(self):
        print(self.graph)

    def draw(self):
        self.graph.visualize()

    def save_json(self):
        fname = input("Имя файла: ")
        self.graph.to_json(fname)
        print("Сохранено.")

    def exit_app(self):
        print("\nЗавершение работы.")
        sys.exit(0)