# data_manager.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional, Dict, Any, List
from config import AppConfig, DataValidationRules

class AppConfig:
    """Базовая конфигурация"""
    MAX_FILE_SIZE_MB = 50
    DISPLAY_DATE_FORMAT = "%d.%m.%Y"

class AdvancedFeatureEngineer:
    """Создание расширенных признаков для временных рядов"""
    
    def __init__(self, country='RU'):
        self.country = country
        self.holidays = self._load_holidays()
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает расширенные признаки из базовых данных"""
        df = df.copy()
        
        # 1. Сначала создаем базовые признаки из date
        df = self._create_basic_features(df)
        
        # 2. Расширенные временные признаки
        df = self._create_temporal_features(df)
        
        # 3. Сезонные и циклические признаки
        df = self._create_seasonal_features(df)
        
        # 4. Праздничные и календарные признаки
        df = self._create_calendar_features(df)
        
        # 5. Статистические признаки
        df = self._create_statistical_features(df)
        
        return df
    
    def _create_basic_features(self, df):
        """Создает базовые признаки из даты"""
        # Убедимся, что date в правильном формате
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Базовые временные признаки
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        return df
    
    def _create_temporal_features(self, df):
        """Расширенные временные признаки"""
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        return df
    
    def _create_seasonal_features(self, df):
        """Сезонные и циклические признаки"""
        # Циклическое кодирование месяцев
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Циклическое кодирование дней недели
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Циклическое кодирование дней года
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Сезоны
        df['season'] = df['month'] % 12 // 3 + 1
        
        return df
    
    def _create_calendar_features(self, df):
        """Календарные признаки для России"""
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Российские праздники
        holiday_dates = [pd.Timestamp(d) for d in self.holidays]
        df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)
        
        # Предпраздничные дни
        pre_holiday_dates = [pd.Timestamp(d) + pd.Timedelta(days=1) for d in self.holidays]
        df['is_pre_holiday'] = df['date'].isin(pre_holiday_dates).astype(int)        
        # Дни после праздников
        post_holiday_dates = [pd.Timestamp(d) - pd.Timedelta(days=1) for d in self.holidays]
        df['is_post_holiday'] = df['date'].isin(post_holiday_dates).astype(int)        
        return df
    
    def _create_statistical_features(self, df):
        """Статистические признаки на основе истории"""
        target = 'total_sales'
        
        # Лаговые признаки
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'sales_lag_{lag}'] = df[target].shift(lag)
        
        # Скользящие статистики
        windows = [7, 14, 30]
        for window in windows:
            df[f'rolling_mean_{window}'] = df[target].rolling(window).mean()
            df[f'rolling_std_{window}'] = df[target].rolling(window).std()
            df[f'rolling_min_{window}'] = df[target].rolling(window).min()
            df[f'rolling_max_{window}'] = df[target].rolling(window).max()
            
            # Отношение к скользящему среднему
            df[f'ratio_to_rolling_mean_{window}'] = df[target] / df[f'rolling_mean_{window}']
        
        # Признаки тренда
        df['sales_trend_7'] = df[target] / df['sales_lag_7']
        df['sales_trend_30'] = df[target] / df['sales_lag_30']
        
        # Волатильность
        df['volatility_14'] = df[target].rolling(14).std() / df[target].rolling(14).mean()
        
        return df
    
    def _load_holidays(self):
        """Загрузка российских праздников"""
        holidays = []
        current_year = datetime.now().year
        
        for year in range(current_year - 2, current_year + 2):
            holidays.extend([
                f"{year}-01-01", f"{year}-01-02", f"{year}-01-03", 
                f"{year}-01-04", f"{year}-01-05", f"{year}-01-06", 
                f"{year}-01-07", f"{year}-01-08",
                f"{year}-02-23", f"{year}-03-08", f"{year}-05-01",
                f"{year}-05-09", f"{year}-06-12", f"{year}-11-04"
            ])
        
        return [datetime.strptime(d, '%Y-%m-%d').date() for d in holidays]

class DataManager:
    """Менеджер для работы с данными"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = DataValidationRules()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.config = AppConfig()
    
    def load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """Загружает данные из CSV файла с валидацией"""
        try:
            self.logger.info(f"Загрузка данных из {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл не найден: {file_path}")
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            if file_size > self.config.MAX_FILE_SIZE_MB:
                raise ValueError(f"Файл слишком большой: {file_size:.1f}MB")
            
            # Загружаем только обязательные колонки
            df = pd.read_csv(file_path, parse_dates=['date'])
            self._validate_raw_data(df)
            
            self.logger.info(f"Успешно загружено {len(df)} записей")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {str(e)}")
            raise
    
    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        """Валидация сырых данных"""
        missing_columns = [col for col in self.validation_rules.REQUIRED_COLUMNS 
                          if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
        
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise ValueError("Колонка 'date' должна содержать даты")
        
        for column in ['quantity', 'price']:
            if column in df.columns:
                min_val, max_val = self.validation_rules.VALUE_RANGES[column]
                if (df[column] < min_val).any() or (df[column] > max_val).any():
                    raise ValueError(f"Значения в колонке '{column}' вне диапазона")
        
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            self.logger.warning(f"Найдено {duplicate_count} дубликатов")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очищает и агрегирует данные с фильтрацией выбросов"""
        try:
            self.logger.info("Начало обработки данных")
            
            df_clean = df.copy()
            initial_count = len(df_clean)
            
            # Базовая очистка
            df_clean = df_clean.drop_duplicates().dropna()
            cleaned_count = initial_count - len(df_clean)
            
            if cleaned_count > 0:
                self.logger.info(f"Удалено дубликатов и пропусков: {cleaned_count}")
            
            # Агрегация по дням
            df_daily = df_clean.groupby('date', as_index=False).agg({
                'quantity': 'sum', 
                'price': 'mean'
            })
            df_daily['total_sales'] = df_daily['quantity'] * df_daily['price']
            
            # ФИЛЬТРАЦИЯ ВЫБРОСОВ - КРИТИЧЕСКИ ВАЖНО
            df_daily = self._remove_sales_outliers(df_daily)
            
            # Создание расширенных признаков
            df_daily = self.feature_engineer.create_advanced_features(df_daily)
            
            # Финальная валидация
            self._validate_processed_data(df_daily)
            
            self.logger.info(f"Обработка завершена. Итоговых записей: {len(df_daily)}")
            return df_daily
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки данных: {str(e)}")
            raise

    def _remove_sales_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Минимальная фильтрация выбросов"""
        try:
            # ВОЗВРАЩАЕМ ВСЕ ДАННЫЕ БЕЗ ФИЛЬТРАЦИИ
            # Пусть модель сама научится работать с вариациями
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка фильтрации: {e}")
            return df
    
    def _validate_processed_data(self, df: pd.DataFrame) -> None:
        """Валидация обработанных данных"""
        if len(df) < 1:
            raise ValueError("Нет данных после обработки")
        
        required_cols = ['date', 'total_sales']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки после обработки: {missing_cols}")
        
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
                'start': df['date'].min().strftime(self.config.DISPLAY_DATE_FORMAT),
                'end': df['date'].max().strftime(self.config.DISPLAY_DATE_FORMAT),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'data_quality': {
                'missing_dates': self._find_missing_dates(df),
                'weekend_ratio': float(df['is_weekend'].mean()) if 'is_weekend' in df else 0
            }
        }
    
    def _find_missing_dates(self, df: pd.DataFrame) -> int:
        """Находит пропущенные даты в последовательности"""
        if len(df) < 2:
            return 0
        
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
        missing_dates = date_range.difference(df['date'])
        return len(missing_dates)