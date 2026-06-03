from abc import ABC, abstractmethod
import pytest
import numpy as np
import sys
import os

repo_path = r'D:\JupyterNotebook\linear_model_task'
sys.path.insert(0, os.path.join(repo_path, 'src'))

# Импортируем тестируемый класс
from linearModel import LinearModel


def test_linear_model_is_abstract():
    """Проверяем, что LinearModel является абстрактным классом"""
    assert issubclass(LinearModel, ABC)


def test_linear_model_has_required_methods():
    """Проверяем наличие требуемых методов"""
    # Обязательные методы
    assert hasattr(LinearModel, "fit")
    assert hasattr(LinearModel, "predict")
    assert hasattr(LinearModel, "score")
    
    # Абстрактные методы
    assert hasattr(LinearModel, "_forward")
    assert hasattr(LinearModel, "_compute_loss")
    assert hasattr(LinearModel, "_compute_gradients")
    assert hasattr(LinearModel, "_compute_metric")


def test_linear_model_cannot_be_instantiated():
    """Проверяем, что нельзя создать экземпляр абстрактного класса"""
    with pytest.raises(TypeError):
        LinearModel()
