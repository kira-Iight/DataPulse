# config.py
import os
from datetime import datetime
from typing import Dict, Any, List

class AppConfig:
    """Конфигурация приложения"""
    
    # Форматы данных
    DATETIME_FORMAT = "%Y-%m-%d"
    DISPLAY_DATE_FORMAT = "%d.%m.%Y"
    
    # Требования к данным
    MIN_DATA_POINTS = 30
    FORECAST_DAYS = 7
    MAX_FILE_SIZE_MB = 50
    
    # Параметры моделей
    MODEL_PARAMS = {
        'simple_average': {
            'window_7_weight': 0.6,
            'window_30_weight': 0.4,
            'confidence_level': 0.88
        }
    }
    
    # Настройки кросс-валидации
    TIME_SERIES_CV = {
        'n_splits': 5,
        'test_size': 7,
        'gap': 0
    }
    
    # Пути
    LOGS_DIR = "logs"
    CACHE_DIR = "cache"
    MODELS_DIR = "models"
    
    # Цветовая схема
    COLORS = {
        'primary': '#2563EB',
        'primary_light': '#3B82F6',
        'secondary': '#64748B',
        'success': '#10B981',
        'warning': '#F59E0B',
        'danger': '#EF4444',
        'dark': '#1E293B',
        'light': '#F8FAFC',
        'background': '#F1F5F9',
        'card': '#FFFFFF',
        'border': '#E2E8F0'
    }
    
    FONTS = {
        'title': ('Segoe UI', 16, 'bold'),
        'subtitle': ('Segoe UI', 12, 'bold'),
        'normal': ('Segoe UI', 10),
        'small': ('Segoe UI', 9),
        'metric': ('Segoe UI', 14, 'bold')
    }

class DataValidationRules:
    """Правила валидации данных"""
    
    REQUIRED_COLUMNS = ['date', 'quantity', 'price']
    DATE_RANGE = {
        'min': datetime(2000, 1, 1),
        'max': datetime(2030, 12, 31)
    }
    VALUE_RANGES = {
        'quantity': (0, 1000000),
        'price': (0, 10000000),
        'total_sales': (0, 100000000)
    }