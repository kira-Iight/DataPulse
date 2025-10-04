# data_manager.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional, Dict, Any

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

class DataManager:
    """Менеджер для работы с данными"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = DataValidationRules()
    
    def load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """Загружает данные из CSV файла с валидацией"""
        try:
            self.logger.info(f"Загрузка данных из {file_path}")
            
            # Проверка существования файла
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл не найден: {file_path}")
            
            # Проверка размера файла
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            if file_size > AppConfig.MAX_FILE_SIZE_MB:
                raise ValueError(f"Файл слишком большой: {file_size:.1f}MB (максимум {AppConfig.MAX_FILE_SIZE_MB}MB)")
            
            # Загрузка данных
            df = pd.read_csv(file_path, parse_dates=['date'])
            
            # Валидация данных
            self._validate_raw_data(df)
            
            self.logger.info(f"Успешно загружено {len(df)} записей")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {str(e)}")
            raise
    
    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        """Валидация сырых данных"""
        # Проверка обязательных колонок
        missing_columns = [col for col in self.validation_rules.REQUIRED_COLUMNS 
                          if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
        
        # Проверка типов данных
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise ValueError("Колонка 'date' должна содержать даты")
        
        # Проверка диапазонов значений
        for column in ['quantity', 'price']:
            if column in df.columns:
                min_val, max_val = self.validation_rules.VALUE_RANGES[column]
                if (df[column] < min_val).any() or (df[column] > max_val).any():
                    raise ValueError(f"Значения в колонке '{column}' вне допустимого диапазона")
        
        # Проверка дубликатов
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            self.logger.warning(f"Найдено {duplicate_count} дубликатов")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очищает и агрегирует данные"""
        try:
            self.logger.info("Начало обработки данных")
            
            # Создаем копию для безопасности
            df_clean = df.copy()
            
            # Удаление дубликатов и обработка пропусков
            initial_count = len(df_clean)
            df_clean = df_clean.drop_duplicates().dropna()
            cleaned_count = initial_count - len(df_clean)
            
            if cleaned_count > 0:
                self.logger.info(f"Удалено записей: {cleaned_count}")
            
            # Агрегация по дням
            df_daily = df_clean.groupby('date', as_index=False).agg({
                'quantity': 'sum', 
                'price': 'mean'
            })
            df_daily['total_sales'] = df_daily['quantity'] * df_daily['price']
            
            # Создание расширенных признаков
            df_daily = self._create_advanced_features(df_daily)
            
            # Финальная валидация
            self._validate_processed_data(df_daily)
            
            self.logger.info(f"Обработка завершена. Итоговых записей: {len(df_daily)}")
            return df_daily[['date', 'total_sales', 'day_of_week', 'month', 
                           'quarter', 'week_of_year', 'is_weekend', 'is_holiday']]
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки данных: {str(e)}")
            raise
    
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает расширенные признаки"""
        df = df.copy()
        
        # Базовые временные признаки
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Признаки сезонности
        df['is_holiday'] = self._identify_holidays(df['date'])
        
        # Лаговые признаки (будут заполнены позже в ML модели)
        for lag in [1, 7, 14, 30]:
            df[f'sales_lag_{lag}'] = df['total_sales'].shift(lag)
        
        # Скользящие статистики
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df['total_sales'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['total_sales'].rolling(window).std()
        
        return df
    
    def _identify_holidays(self, dates: pd.Series) -> pd.Series:
        """Идентифицирует праздничные дни (упрощенная версия)"""
        holidays = []
        for date in dates:
            # Новогодние праздники и основные российские праздники
            is_holiday = (date.month == 1 and date.day <= 8) or \
                        (date.month == 12 and date.day >= 30) or \
                        (date.month == 5 and date.day in [1, 9]) or \
                        (date.month == 3 and date.day == 8) or \
                        (date.month == 11 and date.day == 4) or \
                        (date.month == 2 and date.day == 23)
            holidays.append(is_holiday)
        
        return pd.Series(holidays, index=dates.index)
    
    def _validate_processed_data(self, df: pd.DataFrame) -> None:
        """Валидация обработанных данных"""
        if len(df) < 1:
            raise ValueError("Нет данных после обработки")
        
        # Проверка наличия необходимых колонок
        required_cols = ['date', 'total_sales']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки после обработки: {missing_cols}")
        
        # Проверка корректности дат
        if df['date'].isna().any():
            raise ValueError("Найдены пустые даты")
        
        self.logger.info("Валидация данных пройдена успешно")
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Возвращает статистику данных"""
        if df.empty:
            return {}
        
        sales_data = df['total_sales']
        
        return {
            'total_records': len(df),
            'total_sales': float(sales_data.sum()),
            'avg_daily': float(sales_data.mean()),
            'max_sales': float(sales_data.max()),
            'min_sales': float(sales_data.min()),
            'std_sales': float(sales_data.std()),
            'date_range': {
                'start': df['date'].min().strftime(AppConfig.DISPLAY_DATE_FORMAT),
                'end': df['date'].max().strftime(AppConfig.DISPLAY_DATE_FORMAT),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'data_quality': {
                'missing_dates': self._find_missing_dates(df),
                'weekend_ratio': float(df['is_weekend'].mean())
            }
        }
    
    def _find_missing_dates(self, df: pd.DataFrame) -> int:
        """Находит пропущенные даты в последовательности"""
        if len(df) < 2:
            return 0
        
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
        missing_dates = date_range.difference(df['date'])
        return len(missing_dates)

# Функции для обратной совместимости
def load_data_from_csv(file_path):
    """Функция для обратной совместимости со старым кодом"""
    manager = DataManager()
    return manager.load_data_from_csv(file_path)

def preprocess_data(df):
    """Функция для обратной совместимости со старым кодом"""
    manager = DataManager()
    return manager.preprocess_data(df)