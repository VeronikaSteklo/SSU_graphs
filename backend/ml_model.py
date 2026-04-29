import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
import os


class TrafficPredictor:
    """ML модель для предсказания трафика на ребрах графа"""

    def __init__(self):
        self.model = None
        self.base_weights = {}  # Базовые веса ребер
        self.is_trained = False

    def train(self, edges_data: list):
        X = []
        y = []

        for u, v, base_weight in edges_data:
            self.base_weights[(u, v)] = base_weight

            for hour in range(24):
                for day_type in [0, 1]:
                    for weather in range(3):
                        features = self._extract_features(hour, day_type, weather, base_weight)
                        X.append(features)

                        traffic_factor = self._simulate_traffic_pattern(hour, day_type, weather)
                        actual_weight = base_weight * traffic_factor
                        y.append(actual_weight)

        X = np.array(X)
        y = np.array(y)

        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X, y)
        self.is_trained = True

        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/traffic_model.pkl')
        joblib.dump(self.base_weights, 'models/base_weights.pkl')

    def _extract_features(self, hour: int, day_type: int, weather: int, base_weight: float):
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        return np.array([
            hour_sin,
            hour_cos,
            day_type,
            weather,
            base_weight,
            hour * day_type,
            hour * weather
        ])

    def _simulate_traffic_pattern(self, hour: int, day_type: int, weather: int) -> float:
        factor = 1.0

        if day_type == 0:
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                factor += 0.8
            elif 10 <= hour <= 16:
                factor += 0.3
            elif 0 <= hour <= 5:
                factor -= 0.4
        else:
            if 11 <= hour <= 20:
                factor += 0.4
            elif 0 <= hour <= 7:
                factor -= 0.5

        if weather == 1:
            factor += 0.2
        elif weather == 2:
            factor += 0.4

        return max(0.3, factor)

    def predict_weight(self, u: str, v: str, hour: int, day_type: int = 0, weather: int = 0) -> float:
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        base_weight = self.base_weights.get((u, v), 1.0)
        features = self._extract_features(hour, day_type, weather, base_weight)
        prediction = self.model.predict([features])[0]

        return max(0.1, prediction)

    def predict_all_edges(self, hour: int, day_type: int = 0, weather: int = 0):
        predictions = {}
        for (u, v), base_weight in self.base_weights.items():
            weight = self.predict_weight(u, v, hour, day_type, weather)
            predictions[(u, v)] = {
                'base': base_weight,
                'current': weight,
                'congestion': (weight / base_weight - 1) * 100
            }
        return predictions

    @classmethod
    def load_model(cls):
        instance = cls()
        if os.path.exists('models/traffic_model.pkl'):
            instance.model = joblib.load('models/traffic_model.pkl')
            instance.base_weights = joblib.load('models/base_weights.pkl')
            instance.is_trained = True
        return instance