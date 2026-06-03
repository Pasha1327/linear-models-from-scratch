import numpy as np
from linearModel import LinearModel
from sklearn.metrics import accuracy_score

class LogisticRegression(LinearModel):
    """
    Класс логистической регрессии.
    Реализует абстрактные методы родительского класса.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        super().__init__(learning_rate, n_iterations)
    
    def _forward(self, linear_output):
        """Для логистической регрессии применяем сигмоиду"""
        return self._sigmoid(linear_output)
    
    def _compute_loss(self, y_true, y_pred):
        """Вычисление log loss (бинарная кросс-энтропия)"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def _compute_gradients(self, X, y_true, y_pred):
        """Вычисление градиентов для log loss"""
        n_samples = len(y_true)
        error = y_pred - y_true
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)
        return dw, db
    
    def _compute_metric(self, y_true, y_pred):
        """Accuracy для оценки качества классификации"""
        y_pred_classes = (y_pred > 0.5).astype(int)
        return accuracy_score(y_true, y_pred_classes)
    
    def predict(self, X):
        """Переопределяем predict для возврата классов"""
        probabilities = super().predict(X)
        return (probabilities > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Возвращает вероятности принадлежности к классу 1"""
        return super().predict(X)
    
    def auto_tune(self, X, y, learning_rates=[0.001, 0.01, 0.1], 
                  n_iterations_list=[500, 1000, 2000]):
        """Автоматический подбор гиперпараметров"""
        from sklearn.model_selection import train_test_split
        
        best_score = -float('inf')
        best_params = {}
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        for lr in learning_rates:
            for n_iter in n_iterations_list:
                model = LogisticRegression(learning_rate=lr, n_iterations=n_iter)
                model.fit(X_train, y_train)
                
                score = model.score(X_val, y_val)
                
                if score > best_score:
                    best_score = score
                    best_params = {'learning_rate': lr, 'n_iterations': n_iter}
        
        self.learning_rate = best_params['learning_rate']
        self.n_iterations = best_params['n_iterations']
        self.fit(X, y)
        
        return best_params, best_score

