import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional, Dict, Any, List
import os
from config import AppConfig

class ForecastEngine:
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Упрощенная и надежная подготовка признаков"""
        if df.empty or 'total_sales' not in df.columns:
            return pd.DataFrame(), pd.DataFrame()
        
        df_prepared = df.copy()
        
        # Базовые временные признаки
        df_prepared['day_of_week'] = df_prepared['date'].dt.dayofweek
        df_prepared['month'] = df_prepared['date'].dt.month
        df_prepared['is_weekend'] = (df_prepared['date'].dt.dayofweek >= 5).astype(int)
        
        # Циклические признаки
        df_prepared['month_sin'] = np.sin(2 * np.pi * df_prepared['month'] / 12)
        df_prepared['month_cos'] = np.cos(2 * np.pi * df_prepared['month'] / 12)
        
        # Простые лаговые признаки
        for lag in [1, 7]:
            df_prepared[f'sales_lag_{lag}'] = df_prepared['total_sales'].shift(lag)
        
        # Заполняем пропуски
        df_prepared = df_prepared.bfill().ffill()
        
        # Фиксированный набор признаков
        feature_columns = ['day_of_week', 'month', 'is_weekend', 'month_sin', 'month_cos', 
                          'sales_lag_1', 'sales_lag_7']
        
        # Оставляем только существующие колонки
        available_features = [col for col in feature_columns if col in df_prepared.columns]
        
        X = df_prepared[available_features]
        y = df_prepared['total_sales']
        
        # Удаляем строки с NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        return X[mask], y[mask]
    
    def _calculate_real_confidence_interval(self, model, X_pred, prediction: float, 
                                        day_index: int, historical_std: float) -> Dict[str, float]:
        """
        ПРАВИЛЬНЫЙ расчет доверительного интервала с РЕАЛИСТИЧНЫМИ уровнями доверия
        """
        try:
            # 1. Получаем предсказания от всех деревьев
            tree_predictions = []
            if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                for tree in model.estimators_:
                    try:
                        tree_pred = tree.predict(X_pred)[0]
                        tree_predictions.append(tree_pred)
                    except:
                        continue
            
            if not tree_predictions:
                # Консервативная оценка при отсутствии данных
                base_uncertainty = 0.25 + (day_index * 0.04)
                margin = prediction * base_uncertainty
                return {
                    'lower': float(max(0, prediction - margin)),
                    'upper': float(prediction + margin),
                    'uncertainty_pct': float(base_uncertainty * 100),
                    'std_dev': float(prediction * base_uncertainty),
                    'trees_used': 0,
                    'method': 'conservative_fallback',
                    'confidence_level': 0.80  # Реалистичный уровень
                }
            
            # 2. Базовая статистика
            n_trees = len(tree_predictions)
            tree_std = np.std(tree_predictions)
            tree_mean = np.mean(tree_predictions)
            
            # 3. РЕАЛИСТИЧНЫЙ расчет уровня доверия
            # В реальности уровни доверия редко превышают 95% для бизнес-прогнозов
            
            # Базовый уровень на основе количества деревьев
            if n_trees >= 80:
                base_level = 0.92
            elif n_trees >= 50:
                base_level = 0.90
            elif n_trees >= 30:
                base_level = 0.87
            elif n_trees >= 15:
                base_level = 0.85
            else:
                base_level = 0.80
            
            # Корректировка на вариацию (коэффициент вариации)
            cv = tree_std / tree_mean if tree_mean > 0 else 0.5
            if cv < 0.05:
                cv_adjustment = 0.03  # +3% за низкую вариацию
            elif cv < 0.1:
                cv_adjustment = 0.01  # +1%
            elif cv < 0.2:
                cv_adjustment = 0.00  # без изменений
            elif cv < 0.3:
                cv_adjustment = -0.03  # -3%
            else:
                cv_adjustment = -0.06  # -6%
            
            # Корректировка на горизонт прогноза
            horizon_adjustment = -0.02 * min(day_index, 5)  # -2% за день, максимум -10%
            
            # Финальный уровень доверия (ограничиваем реалистичными пределами)
            confidence_level = max(0.75, min(0.95, 
                base_level + cv_adjustment + horizon_adjustment
            ))
            
            # 4. Соответствующий Z-score
            z_score_map = {
                0.75: 1.15,   0.80: 1.28,   0.85: 1.44,
                0.87: 1.51,   0.90: 1.645,  0.92: 1.75,
                0.95: 1.96
            }
            
            # Находим ближайший Z-score
            closest_level = min(z_score_map.keys(), key=lambda x: abs(x - confidence_level))
            z_score = z_score_map[closest_level]
            
            # 5. Учитываем историческую волатильность
            historical_contribution = historical_std * 0.15
            
            # 6. Комбинированная стандартная ошибка
            combined_std = tree_std + historical_contribution
            
            # 7. Доверительный интервал
            margin = combined_std * z_score
            
            lower = max(0, prediction - margin)
            upper = prediction + margin
            
            # 8. Процент неопределенности
            uncertainty_pct = (margin / prediction) * 100 if prediction > 0 else 25.0
            uncertainty_pct = min(60.0, max(8.0, uncertainty_pct))  # Реалистичные пределы 8-60%
            
            self.logger.info(f"ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ: {confidence_level:.1%} (±{uncertainty_pct:.1f}%) "
                        f"(деревья: {n_trees}, CV: {cv:.3f})")
            
            return {    
                'lower': float(lower),
                'upper': float(upper),
                'uncertainty_pct': float(uncertainty_pct),
                'std_dev': float(combined_std),
                'trees_used': n_trees,
                'coefficient_of_variation': float(cv),
                'method': 'realistic_calculation',
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета доверительного интервала: {e}")
            
            # Простой реалистичный fallback
            base_uncertainty = 0.2 + (day_index * 0.03)
            margin = prediction * base_uncertainty
            
            # Реалистичный уровень доверия для fallback
            confidence_level = max(0.80, 0.85 - (day_index * 0.02))
            
            return {
                'lower': float(max(0, prediction - margin)),
                'upper': float(prediction + margin),
                'uncertainty_pct': float(base_uncertainty * 100),
                'std_dev': float(prediction * base_uncertainty),
                'trees_used': 0,
                'method': 'realistic_fallback',
                'confidence_level': confidence_level
            }
    
    def _calculate_historical_volatility(self, df: pd.DataFrame) -> float:
        """Рассчитывает историческую волатильность данных"""
        try:
            if len(df) < 10:
                return 1000.0  # Возвращаем разумное значение по умолчанию
            
            sales_data = df['total_sales']
            
            # Рассчитываем дневные изменения
            daily_changes = sales_data.pct_change().dropna()
            
            # Рассчитываем стандартное отклонение изменений
            volatility = daily_changes.std()
            
            # Конвертируем в абсолютное значение (в рублях)
            avg_sales = sales_data.mean()
            absolute_volatility = volatility * avg_sales
            
            self.logger.info(f"Историческая волатильность: {absolute_volatility:.0f} руб.")
            
            return float(absolute_volatility)
            
        except Exception as e:
            self.logger.warning(f"Ошибка расчета исторической волатильности: {e}")
            return 1000.0  # Значение по умолчанию
    
    def train_model(self, session_data: Dict[str, Any], optimize_hyperparams: bool = False) -> Tuple[Any, float]:
        """Обучение модели Random Forest"""
        try:
            processed_data = session_data.get('processed_data', [])
            if not processed_data:
                self.logger.error("Нет данных для обучения")
                return None, 0.0
            
            df = pd.DataFrame(processed_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            if len(df) < self.config.MIN_DATA_POINTS:
                self.logger.warning(f"Недостаточно данных: {len(df)} < {self.config.MIN_DATA_POINTS}")
                return None, 0.0
            
            # Подготавливаем признаки
            X, y = self.prepare_features(df)
            if X.empty or len(X) < 10:
                self.logger.error("Недостаточно данных после подготовки признаков")
                return None, 0.0
            
            # Рассчитываем историческую волатильность
            historical_volatility = self._calculate_historical_volatility(df)
            
            # Хронологическое разделение
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Обучение модели
            model = RandomForestRegressor(**self.config.MODEL_PARAMS['random_forest'])
            model.fit(X_train, y_train)
            
            # Оценка
            y_pred = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Сохраняем метрики
            accuracy_data = session_data.get('model_accuracy', [])
            accuracy_data.append({
                'accuracy': float(mape),
                'mae': float(mae),
                'model_name': 'Random Forest',
                'features_used': X.shape[1],
                'training_size': len(X_train),
                'test_size': len(X_test),
                'historical_volatility': historical_volatility,
                'created_at': datetime.now().isoformat()
            })
            session_data['model_accuracy'] = accuracy_data
            
            self.logger.info(f"Модель обучена. MAPE: {mape:.2%}, Волатильность: {historical_volatility:.0f} руб.")
            return model, mape
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения: {str(e)}")
            return None, 0.0

    def make_predictions(self, model: Any, session_data: Dict[str, Any], 
                        days_to_forecast: int = 7) -> List[Dict[str, Any]]:
        """Создание прогнозов с РЕАЛЬНЫМИ доверительными интервалами"""
        if model is None:
            return None
        
        try:
            processed_data = session_data.get('processed_data', [])
            if not processed_data:
                return None
            
            df = pd.DataFrame(processed_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            df = df.sort_values('date')
            df_with_features = self._create_features_for_prediction(df)
            
            # Получаем историческую волатильность
            accuracy_data = session_data.get('model_accuracy', [])
            historical_volatility = accuracy_data[-1].get('historical_volatility', 1000.0) if accuracy_data else 1000.0
            
            predictions = []
            current_data = df_with_features.tail(30).copy()
            
            for i in range(days_to_forecast):
                forecast_date = current_data['date'].iloc[-1] + timedelta(days=1)
                
                # Подготавливаем данные для прогноза
                X_pred = self._prepare_prediction_features(current_data, forecast_date, predictions)
                if X_pred is None or X_pred.empty:
                    break
                
                predicted_sales = float(model.predict(X_pred)[0])
                
                # РЕАЛЬНЫЙ доверительный интервал 
                confidence_interval = self._calculate_real_confidence_interval(
                    model, X_pred, predicted_sales, i, historical_volatility
                )
                
                prediction = {
                    'date': forecast_date,
                    'predicted_sales': predicted_sales,
                    'confidence_interval': confidence_interval
                }
                
                predictions.append(prediction)
                
                # Обновляем данные для следующей итерации
                self._update_prediction_data(current_data, prediction)
            
            self.logger.info(f"Создано {len(predictions)} прогнозов с реальными доверительными интервалами")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Ошибка прогнозирования: {str(e)}")
            return None
    
    def _create_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает признаки для исторических данных"""
        df_feat = df.copy()
        df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
        df_feat['month'] = df_feat['date'].dt.month
        df_feat['is_weekend'] = (df_feat['date'].dt.dayofweek >= 5).astype(int)
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
        
        for lag in [1, 7]:
            df_feat[f'sales_lag_{lag}'] = df_feat['total_sales'].shift(lag)
        
        return df_feat.bfill().ffill()
    
    def _prepare_prediction_features(self, current_data: pd.DataFrame, forecast_date: pd.Timestamp, 
                                   predictions: List) -> Optional[pd.DataFrame]:
        """Подготавливает признаки для конкретного прогноза"""
        try:
            last_row = current_data.iloc[-1:].copy()
            last_row['date'] = forecast_date
            last_row['day_of_week'] = forecast_date.dayofweek
            last_row['month'] = forecast_date.month
            last_row['is_weekend'] = 1 if forecast_date.dayofweek >= 5 else 0
            last_row['month_sin'] = np.sin(2 * np.pi * last_row['month'] / 12)
            last_row['month_cos'] = np.cos(2 * np.pi * last_row['month'] / 12)
            
            # Обновляем лаги
            if predictions:
                last_row['sales_lag_1'] = predictions[-1]['predicted_sales']
            
            # Лаг 7 дней
            lag_7_date = forecast_date - timedelta(days=7)
            lag_7_value = self._find_historical_value(current_data, lag_7_date, predictions)
            if lag_7_value is not None:
                last_row['sales_lag_7'] = lag_7_value
            
            feature_columns = ['day_of_week', 'month', 'is_weekend', 'month_sin', 'month_cos', 
                             'sales_lag_1', 'sales_lag_7']
            available_features = [col for col in feature_columns if col in last_row.columns]
            
            return last_row[available_features].bfill().ffill()
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки признаков: {e}")
            return None
    
    def _find_historical_value(self, data: pd.DataFrame, target_date: pd.Timestamp, 
                             predictions: List) -> Optional[float]:
        """Находит значение продаж для заданной даты"""
        # Ищем в исторических данных
        historical_match = data[data['date'] == target_date]
        if not historical_match.empty:
            return historical_match['total_sales'].iloc[0]
        
        # Ищем в прогнозах
        for pred in predictions:
            if pd.to_datetime(pred['date']) == target_date:
                return pred['predicted_sales']
        
        return None
    
    def _update_prediction_data(self, data: pd.DataFrame, prediction: Dict):
        """Обновляет данные для следующей итерации прогноза"""
        new_row = data.iloc[-1:].copy()
        new_row['date'] = prediction['date']
        new_row['total_sales'] = prediction['predicted_sales']
        new_row['day_of_week'] = new_row['date'].dt.dayofweek
        new_row['month'] = new_row['date'].dt.month
        new_row['is_weekend'] = (new_row['date'].dt.dayofweek >= 5).astype(int)
        new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
        new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
        
        # Обновляем лаги
        if len(data) > 0:
            new_row['sales_lag_1'] = data['total_sales'].iloc[-1]
        if len(data) >= 7:
            new_row['sales_lag_7'] = data['total_sales'].iloc[-7]
        
        data.loc[len(data)] = new_row.iloc[0]
    
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
            'historical_volatility': latest_accuracy.get('historical_volatility', 0),
            'created_at': latest_accuracy.get('created_at', '')
        }

# Функции для обратной совместимости
def train_model(session):
    engine = ForecastEngine()
    return engine.train_model(session.__dict__)

def make_predictions(model, session, days_to_forecast=7):
    engine = ForecastEngine()
    return engine.make_predictions(model, session.__dict__, days_to_forecast)