# ml_engine.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import randint, uniform
import joblib
from datetime import datetime
import hashlib
import logging
from typing import Tuple, Optional, Dict, Any, List
import os

# Базовая конфигурация
class BaseAppConfig:
    """Базовая конфигурация"""
    MODEL_PARAMS = {
        'random_forest': {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 5,
            'learning_rate': 0.1
        }
    }
    
    HYPERPARAM_GRID = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
    
    TIME_SERIES_CV = {
        'n_splits': 5,
        'test_size': 7,
        'gap': 0
    }
    
    MIN_DATA_POINTS = 30

class ModelFactory:
    """Фабрика для создания различных ML моделей"""
    
    def __init__(self, config: BaseAppConfig = None):
        self.config = config or BaseAppConfig()
    
    def create_model(self, model_type: str, **kwargs):
        """Создает модель по типу"""
        models = {
            'random_forest': self._create_random_forest,
            'gradient_boosting': self._create_gradient_boosting
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type](**kwargs)
    
    def _create_random_forest(self, **kwargs):
        params = {**self.config.MODEL_PARAMS['random_forest'], **kwargs}
        return RandomForestRegressor(**params)
    
    def _create_gradient_boosting(self, **kwargs):
        params = {**self.config.MODEL_PARAMS['gradient_boosting'], **kwargs}
        return GradientBoostingRegressor(**params)

class HyperparameterOptimizer:
    """Оптимизатор гиперпараметров"""
    
    def __init__(self, config: BaseAppConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize_model(self, model_type: str, X, y, cv_folds=5, n_iter=20):
        """Оптимизация гиперпараметров модели"""
        if model_type not in self.config.HYPERPARAM_GRID:
            self.logger.warning(f"Нет конфигурации оптимизации для {model_type}")
            return None, {}
        
        model_factory = ModelFactory(self.config)
        base_model = model_factory.create_model(model_type)
        
        param_distributions = self.config.HYPERPARAM_GRID[model_type]
        
        search = RandomizedSearchCV(
            base_model, param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.logger.info(f"Запуск оптимизации для {model_type}...")
        search.fit(X, y)
        
        self.logger.info(f"Лучшие параметры: {search.best_params_}")
        self.logger.info(f"Лучший score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_

class TimeSeriesValidator:
    """Валидатор для временных рядов"""
    
    def __init__(self, config: BaseAppConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def time_series_cross_validate(self, model, X, y):
        """Кросс-валидация с учетом временного порядка"""
        tscv = TimeSeriesSplit(
            n_splits=self.config.TIME_SERIES_CV['n_splits'],
            test_size=self.config.TIME_SERIES_CV['test_size'],
            gap=self.config.TIME_SERIES_CV['gap']
        )
        
        scores = []
        feature_importance = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mape = self._calculate_mape(y_test, y_pred)
            mae = self._calculate_mae(y_test, y_pred)
            
            scores.append({
                'fold': fold,
                'mape': mape,
                'mae': mae,
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
            
            if hasattr(model, 'feature_importances_'):
                feature_importance.append(model.feature_importances_)
        
        return {
            'scores': scores,
            'mean_mape': np.mean([s['mape'] for s in scores]),
            'std_mape': np.std([s['mape'] for s in scores]),
            'mean_mae': np.mean([s['mae'] for s in scores]),
            'feature_importance': np.mean(feature_importance, axis=0) if feature_importance else None
        }
    
    def _calculate_mape(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    
    def _calculate_mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

class ForecastEngine:
    """Движок для прогнозирования с расширенными возможностями"""
    
    def __init__(self, config: BaseAppConfig = None):
        self.config = config or BaseAppConfig()
        self.logger = logging.getLogger(__name__)
        self.model_factory = ModelFactory(self.config)
        self.validator = TimeSeriesValidator(self.config)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Создает необходимые директории"""
        os.makedirs("cache", exist_ok=True)
        os.makedirs("models", exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Подготавливает признаки для модели"""
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        df_prepared = df.copy()
        
        # Выбираем только числовые колонки, исключая целевую переменную
        feature_columns = [col for col in df_prepared.select_dtypes(include=[np.number]).columns 
                          if col != 'total_sales' and not col.startswith('date')]
        
        # Удаляем колонки с большим количеством пропусков
        feature_columns = [col for col in feature_columns 
                          if df_prepared[col].notna().sum() > len(df_prepared) * 0.5]
        
        # Заполняем пропущенные значения
        for col in feature_columns:
                df_prepared[col] = df_prepared[col].bfill().ffill()        
        X = df_prepared[feature_columns]
        y = df_prepared['total_sales']
        
        self.logger.info(f"Используется признаков: {len(feature_columns)}")
        return X, y
    
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Создает хэш данных для кэширования"""
        data_str = df[['date', 'total_sales']].to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def train_model(self, session_data: Dict[str, Any], optimize_hyperparams: bool = False) -> Tuple[Any, float]:
        """Обучает модель Random Forest"""
        try:
            processed_data = session_data.get('processed_data', [])
            if not processed_data:
                self.logger.error("Нет данных для обучения")
                return None, None
            
            df = pd.DataFrame(processed_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            if len(df) < self.config.MIN_DATA_POINTS:
                self.logger.warning(f"Мало данных для обучения: {len(df)}")
                return None, None
            
            # Подготавливаем признаки
            X, y = self.prepare_features(df)
            if X.empty:
                self.logger.error("Нет данных после подготовки признаков")
                return None, None
            
            # Создаем хэш для кэширования
            data_hash = self._get_data_hash(df)
            model_path = os.path.join("models", f"model_{data_hash}.joblib")
            
            # Проверяем кэш
            if os.path.exists(model_path):
                self.logger.info("Загрузка модели из кэша")
                model = joblib.load(model_path)
                
                # Оценка модели
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                y_pred = model.predict(X_test)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                self.logger.info(f"Модель загружена из кэша. MAPE: {mape:.3f}")
                return model, mape
            
            # Обучение новой модели
            self.logger.info(f"Обучение новой модели: Random Forest")
            model = self.model_factory.create_model('random_forest')
            model.fit(X, y)
            
            # Кросс-валидация во времени
            cv_results = self.validator.time_series_cross_validate(model, X, y)
            mean_mape = cv_results['mean_mape']
            
            # Сохраняем модель
            joblib.dump(model, model_path)
            
            # Сохраняем метрики
            accuracy_data = session_data.get('model_accuracy', [])
            accuracy_data.append({
                'accuracy': float(mean_mape / 100),  # Конвертируем в долю
                'mae': float(cv_results['mean_mae']),
                'model_name': 'Random Forest',
                'features_used': X.shape[1],
                'training_size': len(X),
                'created_at': datetime.now().isoformat(),
                'cv_mean_mape': float(mean_mape),
                'cv_std_mape': float(cv_results['std_mape'])
            })
            session_data['model_accuracy'] = accuracy_data
            
            self.logger.info(f"✅ Модель обучена. MAPE: {mean_mape:.2f}%")
            return model, mean_mape / 100  # Возвращаем в долях для совместимости
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {str(e)}")
            return None, None

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
            
            # Используем последние доступные данные для прогноза
            last_data = df.iloc[-1:].copy()
            last_date = last_data['date'].iloc[0]
            
            predictions = []
            
            for i in range(days_to_forecast):
                forecast_date = last_date + pd.Timedelta(days=i + 1)
                
                # Создаем данные для прогноза
                forecast_data = last_data.copy()
                forecast_data['date'] = forecast_date
                
                # Обновляем временные признаки
                forecast_data = self._update_temporal_features(forecast_data, forecast_date)
                
                # Подготавливаем фичи
                X_pred, _ = self.prepare_features(forecast_data)
                
                if not X_pred.empty:
                    predicted_sales = float(model.predict(X_pred)[0])
                    
                    predictions.append({
                        'date': forecast_date,
                        'predicted_sales': predicted_sales,
                        'confidence_interval': {
                            'lower': predicted_sales * 0.8,
                            'upper': predicted_sales * 1.2
                        }
                    })
            
            self.logger.info(f"Создано {len(predictions)} прогнозов")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Ошибка создания прогноза: {str(e)}")
            return None
    
    def _update_temporal_features(self, df: pd.DataFrame, date: datetime) -> pd.DataFrame:
        """Обновляет временные признаки для прогнозируемой даты"""
        df = df.copy()
        
        # Базовые временные признаки
        df['day_of_week'] = date.dayofweek
        df['month'] = date.month
        df['quarter'] = date.quarter
        df['week_of_year'] = date.isocalendar().week
        df['is_weekend'] = 1 if date.dayofweek in [5, 6] else 0
        df['day_of_year'] = date.dayofyear
        df['is_month_start'] = 1 if date.day == 1 else 0
        df['is_month_end'] = 1 if date == date.replace(day=28) + pd.Timedelta(days=4) - pd.Timedelta(days=1) else 0
        
        # Сезонные признаки
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
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
            'created_at': latest_accuracy.get('created_at', ''),
            'best_params': latest_accuracy.get('best_params', {}),
            'cv_mean_mape': latest_accuracy.get('cv_mean_mape', 0),
            'cv_std_mape': latest_accuracy.get('cv_std_mape', 0)
        }

# Функции для обратной совместимости
def train_model(session):
    engine = ForecastEngine()
    return engine.train_model(session.__dict__)

def make_predictions(model, session, days_to_forecast=7):
    engine = ForecastEngine()
    return engine.make_predictions(model, session.__dict__, days_to_forecast)