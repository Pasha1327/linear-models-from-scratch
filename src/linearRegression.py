import numpy as np
from linearModel import LinearModel
from sklearn.metrics import r2_score

class LinearRegression(LinearModel):
    """
    Класс линейной регрессии.
    Реализует абстрактные методы родительского класса.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        super().__init__(learning_rate, n_iterations)
    
    def _forward(self, linear_output):
        """Для линейной регрессии возвращаем линейную комбинацию"""
        return linear_output
    
    def _compute_loss(self, y_true, y_pred):
        """Вычисление MSE (Mean Squared Error)"""
        return np.mean((y_true - y_pred) ** 2)
    
    def _compute_gradients(self, X, y_true, y_pred):
        """Вычисление градиентов для MSE"""
        n_samples = len(y_true)
        dw = (-2 / n_samples) * np.dot(X.T, (y_true - y_pred))
        db = (-2 / n_samples) * np.sum(y_true - y_pred)
        return dw, db
    
    def _compute_metric(self, y_true, y_pred):
        """R² score для оценки качества регрессии"""
        return r2_score(y_true, y_pred)
    
    def auto_tune(self, X, y, learning_rates=[0.001, 0.01, 0.1], 
                  n_iterations_list=[500, 1000, 2000]):
        """
        Автоматический подбор гиперпараметров
        """
        from sklearn.model_selection import train_test_split
        
        best_score = -float('inf')
        best_params = {}
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for lr in learning_rates:
            for n_iter in n_iterations_list:
                model = LinearRegression(learning_rate=lr, n_iterations=n_iter)
                model.fit(X_train, y_train)
                
                score = model.score(X_val, y_val)
                
                if score > best_score:
                    best_score = score
                    best_params = {'learning_rate': lr, 'n_iterations': n_iter}
        
        self.learning_rate = best_params['learning_rate']
        self.n_iterations = best_params['n_iterations']
        self.fit(X, y)
        
        return best_params, best_score

