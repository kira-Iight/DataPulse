# config.py
import os
from datetime import datetime

class AppConfig:
    """Конфигурация приложения"""
    
    # Форматы данных
    DATETIME_FORMAT = "%Y-%m-%d"
    DISPLAY_DATE_FORMAT = "%d.%m.%Y"
    
    # Требования к данным
    MIN_DATA_POINTS = 30
    FORECAST_DAYS = 7
    MAX_FILE_SIZE_MB = 50
    
    # Параметры модели
    MODEL_PARAMS = {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1
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