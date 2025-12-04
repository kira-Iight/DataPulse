import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional, Dict, Any, List

# Импорт конфигурации приложения
from config import AppConfig, DataValidationRules

class AdvancedFeatureEngineer:
    """Класс для создания расширенных признаков временных рядов.
    Генерирует дополнительные признаки из дат для улучшения
    качества прогнозирования модели машинного обучения.
    """
    
    def __init__(self, country='RU'):
        """Инициализация инженера признаков. """
        self.country = country
        self.holidays = self._load_holidays()  # Загрузка праздничных дней
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Основной метод создания расширенных признаков"""
        # Создание копии данных для предотвращения модификации оригинала
        df = df.copy()
        
        # Последовательное создание различных типов признаков
        df = self._create_basic_features(df)      # Базовые временные признаки
        df = self._create_temporal_features(df)   # Расширенные временные признаки
        df = self._create_seasonal_features(df)   # Сезонные и циклические признаки
        df = self._create_calendar_features(df)   # Календарные признаки (праздники)
        df = self._create_statistical_features(df) # Статистические признаки
        
        return df
    
    def _create_basic_features(self, df):
        """Создание базовых временных признаков из даты."""
        # Проверка и преобразование формата даты при необходимости
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Извлечение базовых компонентов даты
        df['day_of_week'] = df['date'].dt.dayofweek  # День недели (0=понедельник)
        df['month'] = df['date'].dt.month            # Месяц (1-12)
        df['year'] = df['date'].dt.year              # Год
        
        return df
    
    def _create_temporal_features(self, df):
        """Создание расширенных временных признаков."""
        # Квартал года
        df['quarter'] = df['date'].dt.quarter
        
        # Порядковый номер дня в году
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Номер недели в году (ISO формат)
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Флаги для специальных дней
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)     # Первый день месяца
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)         # Последний день месяца
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int) # Первый день квартала
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)     # Последний день квартала
        
        return df
    
    def _create_seasonal_features(self, df):
        """Создание сезонных и циклических признаков."""
        # Циклическое кодирование месяцев: sin(2π*месяц/12) и cos(2π*месяц/12)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Циклическое кодирование дней недели: sin(2π*день/7) и cos(2π*день/7)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Циклическое кодирование дней года: sin(2π*день/365) и cos(2π*день/365)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Определение сезона года (1-зима, 2-весна, 3-лето, 4-осень)
        df['season'] = df['month'] % 12 // 3 + 1
        
        return df
    
    def _create_calendar_features(self, df):
        """Создание календарных признаков для России."""
        # Определение выходных дней (суббота=5, воскресенье=6)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Определение праздничных дней
        holiday_dates = [pd.Timestamp(d) for d in self.holidays]
        df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)
        
        # Определение предпраздничных дней (день перед праздником)
        pre_holiday_dates = [pd.Timestamp(d) + pd.Timedelta(days=1) for d in self.holidays]
        df['is_pre_holiday'] = df['date'].isin(pre_holiday_dates).astype(int)
        
        # Определение дней после праздников (день после праздника)
        post_holiday_dates = [pd.Timestamp(d) - pd.Timedelta(days=1) for d in self.holidays]
        df['is_post_holiday'] = df['date'].isin(post_holiday_dates).astype(int)
        
        return df
    
    def _create_statistical_features(self, df):
        """Создание статистических признаков на основе истории продаж."""
        target = 'total_sales'  # Целевая переменная для анализа
        
        # Лаговые признаки: значения продаж за предыдущие дни
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'sales_lag_{lag}'] = df[target].shift(lag)
        
        # Скользящие статистики: среднее, стандартное отклонение, минимум, максимум
        windows = [7, 14, 30]  # Размеры окон для скользящих статистик
        for window in windows:
            # Скользящее среднее
            df[f'rolling_mean_{window}'] = df[target].rolling(window).mean()
            
            # Скользящее стандартное отклонение
            df[f'rolling_std_{window}'] = df[target].rolling(window).std()
            
            # Скользящий минимум
            df[f'rolling_min_{window}'] = df[target].rolling(window).min()
            
            # Скользящий максимум
            df[f'rolling_max_{window}'] = df[target].rolling(window).max()
            
            # Отношение текущего значения к скользящему среднему
            df[f'ratio_to_rolling_mean_{window}'] = df[target] / df[f'rolling_mean_{window}']
        
        # Признаки тренда: отношение текущих продаж к продажам N дней назад
        df['sales_trend_7'] = df[target] / df['sales_lag_7']   # Тренд за неделю
        df['sales_trend_30'] = df[target] / df['sales_lag_30'] # Тренд за месяц
        
        # Волатильность: отношение стандартного отклонения к среднему за 14 дней
        df['volatility_14'] = df[target].rolling(14).std() / df[target].rolling(14).mean()
        
        return df
    
    def _load_holidays(self):
        """Загрузка российских праздничных дней."""
        holidays = []
        current_year = datetime.now().year
        
        # Генерация праздников для диапазона лет (от current_year-2 до current_year+1)
        for year in range(current_year - 2, current_year + 2):
            # Основные российские праздники
            holidays.extend([
                f"{year}-01-01", f"{year}-01-02", f"{year}-01-03",  # Новогодние каникулы
                f"{year}-01-04", f"{year}-01-05", f"{year}-01-06", 
                f"{year}-01-07", f"{year}-01-08",                    # Рождество
                f"{year}-02-23",                                     # День защитника Отечества
                f"{year}-03-08",                                     # Международный женский день
                f"{year}-05-01",                                     # Праздник Весны и Труда
                f"{year}-05-09",                                     # День Победы
                f"{year}-06-12",                                     # День России
                f"{year}-11-04"                                      # День народного единства
            ])
        
        # Преобразование строк в объекты datetime.date
        return [datetime.strptime(d, '%Y-%m-%d').date() for d in holidays]

class DataManager:
    """Основной класс для управления данными приложения.
    Отвечает за загрузку, валидацию, очистку и преобразование данных."""
    
    def __init__(self):
        """Инициализация менеджера данных."""
        self.logger = logging.getLogger(__name__)           # Логгер для отслеживания операций
        self.validation_rules = DataValidationRules()       # Правила валидации данных
        self.feature_engineer = AdvancedFeatureEngineer()   # Инженер признаков
        self.config = AppConfig()                           # Конфигурация приложения
    
    def load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """Загрузка данных из CSV файла с полной валидацией."""
        try:
            self.logger.info(f"Загрузка данных из {file_path}")
            
            # Проверка существования файла
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл не найден: {file_path}")
            
            # Проверка размера файла
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Размер в МБ
            if file_size > self.config.MAX_FILE_SIZE_MB:
                raise ValueError(f"Файл слишком большой: {file_size:.1f}MB")
            
            # Загрузка данных из CSV с автоматическим парсингом дат
            df = pd.read_csv(file_path, parse_dates=['date'])
            
            # Валидация загруженных данных
            self._validate_raw_data(df)
            
            self.logger.info(f"Успешно загружено {len(df)} записей")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {str(e)}")
            raise  # Проброс исключения для обработки на верхнем уровне
    
    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        """Валидация сырых данных после загрузки."""
        # Проверка наличия обязательных колонок
        missing_columns = [col for col in self.validation_rules.REQUIRED_COLUMNS 
                          if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
        
        # Проверка формата колонки с датами
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise ValueError("Колонка 'date' должна содержать даты")
        
        # Проверка диапазонов значений для числовых колонок
        for column in ['quantity', 'price']:
            if column in df.columns:
                min_val, max_val = self.validation_rules.VALUE_RANGES[column]
                if (df[column] < min_val).any() or (df[column] > max_val).any():
                    raise ValueError(f"Значения в колонке '{column}' вне диапазона")
        
        # Проверка наличия дубликатов
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            self.logger.warning(f"Найдено {duplicate_count} дубликатов")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предобработка и очистка данных."""
        try:
            self.logger.info("Начало обработки данных")
            
            # Создание копии для предотвращения модификации оригинала
            df_clean = df.copy()
            initial_count = len(df_clean)
            
            # Базовая очистка: удаление дубликатов и строк с пропусками
            df_clean = df_clean.drop_duplicates().dropna()
            cleaned_count = initial_count - len(df_clean)
            
            if cleaned_count > 0:
                self.logger.info(f"Удалено дубликатов и пропусков: {cleaned_count}")
            
            # Агрегация данных по дням
            df_daily = df_clean.groupby('date', as_index=False).agg({
                'quantity': 'sum',    # Суммарное количество товара за день
                'price': 'mean'       # Средняя цена товара за день
            })
            
            # Расчет общей выручки за день
            df_daily['total_sales'] = df_daily['quantity'] * df_daily['price']
            
            # Создание расширенных признаков для улучшения прогнозирования
            df_daily = self.feature_engineer.create_advanced_features(df_daily)
            
            # Финальная валидация обработанных данных
            self._validate_processed_data(df_daily)
            
            self.logger.info(f"Обработка завершена. Итоговых записей: {len(df_daily)}")
            return df_daily
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки данных: {str(e)}")
            raise
    
    def _validate_processed_data(self, df: pd.DataFrame) -> None:
        """Валидация данных после предобработки."""
        # Проверка наличия данных
        if len(df) < 1:
            raise ValueError("Нет данных после обработки")
        
        # Проверка наличия обязательных колонок после обработки
        required_cols = ['date', 'total_sales']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки после обработки: {missing_cols}")
        
        # Проверка наличия пустых дат
        if df['date'].isna().any():
            raise ValueError("Найдены пустые даты")
        
        self.logger.info("Валидация данных пройдена успешно")
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Получение статистики по данным продаж."""
        if df.empty:
            return {}  # Возврат пустого словаря если данных нет
        
        # Основные статистики по продажам
        sales_data = df['total_sales']
        
        return {
            'total_records': len(df),                       # Общее количество записей
            'total_sales': float(sales_data.sum()),         # Суммарная выручка за период
            'avg_daily': float(sales_data.mean()),          # Средняя дневная выручка
            'max_sales': float(sales_data.max()),           # Максимальная дневная выручка
            'min_sales': float(sales_data.min()),           # Минимальная дневная выручка
            'std_sales': float(sales_data.std()),           # Стандартное отклонение выручки
            
            # Информация о периоде данных
            'date_range': {
                'start': df['date'].min().strftime(self.config.DISPLAY_DATE_FORMAT),
                'end': df['date'].max().strftime(self.config.DISPLAY_DATE_FORMAT),
                'days': (df['date'].max() - df['date'].min()).days
            },
            
            # Показатели качества данных
            'data_quality': {
                'missing_dates': self._find_missing_dates(df),  # Количество пропущенных дат
                'weekend_ratio': float(df['is_weekend'].mean()) if 'is_weekend' in df else 0
            }
        }
    
    def _find_missing_dates(self, df: pd.DataFrame) -> int:
        """Поиск пропущенных дат в последовательности."""
        # Для поиска пропусков нужно минимум 2 даты
        if len(df) < 2:
            return 0
        
        # Создание полного диапазона дат от минимальной до максимальной
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
        
        # Поиск дат, которые есть в полном диапазоне, но отсутствуют в данных
        missing_dates = date_range.difference(df['date'])
        
        return len(missing_dates)