from abc import ABC, abstractmethod
import numpy as np

class LinearModel(ABC):
    """
    Абстрактный базовый класс для линейных моделей.
    Определяет общий интерфейс для всех линейных моделей.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Инициализация модели
        
        Parameters:
        learning_rate (float): Скорость обучения
        n_iterations (int): Количество итераций обучения
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _initialize_parameters(self, n_features):
        """Инициализация весов и смещения"""
        self.weights = np.random.randn(n_features)
        self.bias = 0
    
    def _sigmoid(self, z):
        """Сигмоидная функция для логистической регрессии"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def fit(self, X, y):
        """
        Обучение модели на данных
        
        Parameters:
        X (array): Матрица признаков
        y (array): Вектор целевых значений
        """
        n_samples, n_features = X.shape
        
        # Инициализация параметров
        self._initialize_parameters(n_features)
        
        # Градиентный спуск
        for iteration in range(self.n_iterations):
            # Прямое распространение
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self._forward(linear_output)
            
            # Вычисление потерь
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Вычисление градиентов
            dw, db = self._compute_gradients(X, y, y_pred)
            
            # Обновление параметров
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """
        Предсказание на новых данных
        
        Parameters:
        X (array): Матрица признаков
        
        Returns:
        array: Предсказания модели
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self._forward(linear_output)
    
    def score(self, X, y):
        """
        Оценка качества модели
        
        Parameters:
        X (array): Матрица признаков
        y (array): Истинные значения
        
        Returns:
        float: Метрика качества
        """
        y_pred = self.predict(X)
        return self._compute_metric(y, y_pred)
    
    # Абстрактные методы - должны быть реализованы в дочерних классах
    @abstractmethod
    def _forward(self, linear_output):
        """Прямое распространение - преобразование линейной комбинации"""
        pass
    
    @abstractmethod
    def _compute_loss(self, y_true, y_pred):
        """Вычисление функции потерь"""
        pass
    
    @abstractmethod
    def _compute_gradients(self, X, y_true, y_pred):
        """Вычисление градиентов для обновления параметров"""
        pass
    
    @abstractmethod
    def _compute_metric(self, y_true, y_pred):
        """Вычисление метрики качества модели"""
        pass
