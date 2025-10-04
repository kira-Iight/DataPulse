# ml_engine.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib
from datetime import datetime
import hashlib
import logging
from typing import Tuple, Optional, Dict, Any, List
from functools import lru_cache
import os

class ForecastEngine:
    """Движок для прогнозирования с кэшированием"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Создает необходимые директории"""
        os.makedirs("cache", exist_ok=True)
        os.makedirs("models", exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготавливает признаки для модели с расширенными фичами"""
        if df.empty:
            return df
        
        df_prepared = df.copy()
        
        # Создаем расширенные временные признаки
        df_prepared['day_of_month'] = df_prepared['date'].dt.day
        df_prepared['day_of_year'] = df_prepared['date'].dt.dayofyear
        df_prepared['is_month_start'] = df_prepared['date'].dt.is_month_start.astype(int)
        df_prepared['is_month_end'] = df_prepared['date'].dt.is_month_end.astype(int)
        
        # Тригонометрические признаки для цикличности
        df_prepared['month_sin'] = np.sin(2 * np.pi * df_prepared['month'] / 12)
        df_prepared['month_cos'] = np.cos(2 * np.pi * df_prepared['month'] / 12)
        df_prepared['day_of_week_sin'] = np.sin(2 * np.pi * df_prepared['day_of_week'] / 7)
        df_prepared['day_of_week_cos'] = np.cos(2 * np.pi * df_prepared['day_of_week'] / 7)
        
        # Заполняем лаговые признаки
        for lag in [1, 7, 14, 30]:
            col_name = f'sales_lag_{lag}'
            if col_name not in df_prepared.columns:
                df_prepared[col_name] = df_prepared['total_sales'].shift(lag)
        
        # Заполняем скользящие статистики
        for window in [7, 14, 30]:
            mean_col = f'rolling_mean_{window}'
            std_col = f'rolling_std_{window}'
            if mean_col not in df_prepared.columns:
                df_prepared[mean_col] = df_prepared['total_sales'].rolling(window).mean()
            if std_col not in df_prepared.columns:
                df_prepared[std_col] = df_prepared['total_sales'].rolling(window).std()
        
        # Заполняем пропущенные значения
        numeric_columns = df_prepared.select_dtypes(include=[np.number]).columns
        df_prepared[numeric_columns] = df_prepared[numeric_columns].fillna(method='bfill').fillna(method='ffill')
        
        self.logger.info(f"Подготовлено признаков: {len(numeric_columns)}")
        return df_prepared
    
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Создает хэш данных для кэширования"""
        data_str = df[['date', 'total_sales']].to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def train_model(self, session_data: Dict[str, Any]) -> Tuple[Any, float]:
        """Обучает модель машинного обучения с кэшированием"""
        try:
            processed_data = session_data.get('processed_data', [])
            if not processed_data:
                self.logger.error("Нет данных для обучения")
                return None, None
            
            # Конвертируем данные в DataFrame
            df = pd.DataFrame(processed_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Проверяем достаточно ли данных
            if len(df) < 30:
                self.logger.warning(f"Мало данных для обучения: {len(df)} < 30")
                return None, None
            
            # Подготавливаем признаки
            df_prepared = self.prepare_features(df)
            if df_prepared.empty:
                self.logger.error("Нет данных после подготовки признаков")
                return None, None
            
            # Создаем хэш для кэширования
            data_hash = self._get_data_hash(df)
            cache_key = f"{data_hash}_100"  # n_estimators = 100
            model_path = os.path.join("models", f"model_{cache_key}.joblib")
            
            # Проверяем кэш
            if os.path.exists(model_path):
                self.logger.info("Загрузка модели из кэша")
                model = joblib.load(model_path)
                
                # Для простоты пересчитываем метрики
                X = self._get_feature_matrix(df_prepared)
                y = df_prepared['total_sales']
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                y_pred = model.predict(X_test)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                self.logger.info(f"Модель загружена из кэша. MAPE: {mape:.3f}")
                return model, mape
            
            # Обучаем новую модель
            self.logger.info("Обучение новой модели")
            X = self._get_feature_matrix(df_prepared)
            y = df_prepared['total_sales']
            
            # Разделяем данные
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False, random_state=42
            )
            
            # Создаем и обучаем модель
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1
            )
            model.fit(X_train, y_train)
            
            # Оцениваем модель
            y_pred = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Сохраняем модель
            joblib.dump(model, model_path)
            
            # Сохраняем метрики
            accuracy_data = session_data.get('model_accuracy', [])
            accuracy_data.append({
                'accuracy': float(mape),
                'mae': float(mae),
                'model_name': 'RandomForest',
                'features_used': X.shape[1],
                'training_size': len(X_train),
                'created_at': datetime.now().isoformat(),
                'cache_key': cache_key
            })
            session_data['model_accuracy'] = accuracy_data
            
            self.logger.info(f"✅ Модель обучена и сохранена. MAPE: {mape:.3f}, MAE: {mae:.2f}")
            return model, mape
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {str(e)}")
            return None, None
    
    def _get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает матрицу признаков для обучения"""
        # Выбираем только числовые колонки, исключая целевую переменную
        feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                          if col != 'total_sales' and not col.startswith('date')]
        
        # Удаляем колонки с большим количеством пропусков
        feature_columns = [col for col in feature_columns 
                          if df[col].notna().sum() > len(df) * 0.8]
        
        self.logger.info(f"Используется признаков: {len(feature_columns)}")
        return df[feature_columns]
    
    def make_predictions(self, model: Any, session_data: Dict[str, Any], 
                        days_to_forecast: int = 7) -> List[Dict[str, Any]]:
        """Делает прогноз на будущие даты"""
        if model is None:
            self.logger.error("Модель не обучена")
            return None
        
        try:
            processed_data = session_data.get('processed_data', [])
            if not processed_data:
                self.logger.error("Нет данных для прогноза")
                return None
            
            df = pd.DataFrame(processed_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            df_prepared = self.prepare_features(df)
            if df_prepared.empty:
                self.logger.error("Недостаточно данных для прогноза")
                return None
            
            # Получаем последнюю дату
            last_date = df_prepared['date'].iloc[-1]
            predictions = []
            
            # Используем последние доступные данные для прогноза
            current_data = df_prepared.iloc[-1:].copy()
            
            for i in range(days_to_forecast):
                forecast_date = last_date + pd.Timedelta(days=i + 1)
                
                # Обновляем временные признаки для прогнозируемой даты
                current_data['date'] = forecast_date
                current_data['day_of_week'] = forecast_date.dayofweek
                current_data['month'] = forecast_date.month
                current_data['quarter'] = forecast_date.quarter
                current_data['week_of_year'] = forecast_date.isocalendar().week
                current_data['is_weekend'] = 1 if forecast_date.dayofweek in [5, 6] else 0
                current_data['day_of_month'] = forecast_date.day
                current_data['day_of_year'] = forecast_date.dayofyear
                current_data['is_month_start'] = 1 if forecast_date.day == 1 else 0
                current_data['is_month_end'] = 1 if forecast_date == forecast_date.replace(day=28) + pd.Timedelta(days=4) - pd.Timedelta(days=1) else 0
                
                # Тригонометрические признаки
                current_data['month_sin'] = np.sin(2 * np.pi * current_data['month'] / 12)
                current_data['month_cos'] = np.cos(2 * np.pi * current_data['month'] / 12)
                current_data['day_of_week_sin'] = np.sin(2 * np.pi * current_data['day_of_week'] / 7)
                current_data['day_of_week_cos'] = np.cos(2 * np.pi * current_data['day_of_week'] / 7)
                
                # Подготавливаем фичи для предсказания
                X_pred = self._get_feature_matrix(current_data)
                
                # Делаем предсказание
                predicted_sales = float(model.predict(X_pred)[0])
                
                predictions.append({
                    'date': forecast_date,
                    'predicted_sales': predicted_sales,
                    'confidence_interval': {
                        'lower': predicted_sales * 0.8,
                        'upper': predicted_sales * 1.2
                    }
                })
                
                # Обновляем лаговые признаки для следующей итерации
                current_data['sales_lag_1'] = predicted_sales
            
            self.logger.info(f"✅ Создано {len(predictions)} прогнозов")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Ошибка создания прогноза: {str(e)}")
            return None
    
    def get_model_metrics(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Возвращает метрики модели"""
        accuracy_data = session_data.get('model_accuracy', [])
        if not accuracy_data:
            return {}
        
        latest_accuracy = accuracy_data[-1]
        return {
            'accuracy_percent': (1 - latest_accuracy['accuracy']) * 100,
            'mae': latest_accuracy.get('mae', 0),
            'model_name': latest_accuracy.get('model_name', 'Unknown'),
            'features_used': latest_accuracy.get('features_used', 0),
            'training_size': latest_accuracy.get('training_size', 0),
            'created_at': latest_accuracy.get('created_at', '')
        }

# Функции для обратной совместимости
def train_model(session):
    """Функция для обратной совместимости"""
    engine = ForecastEngine()
    return engine.train_model(session.__dict__)

def make_predictions(model, session, days_to_forecast=7):
    """Функция для обратной совместимости"""
    engine = ForecastEngine()
    return engine.make_predictions(model, session.__dict__, days_to_forecast)