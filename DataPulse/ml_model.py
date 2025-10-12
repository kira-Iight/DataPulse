import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime
from loguru import logger

def load_processed_data(session):
    """Загружает обработанные данные из сессии."""
    processed_data = session.get('processed_data', [])
    if not processed_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(processed_data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

def get_recent_data(df, days=30):
    """Возвращает только последние N дней данных"""
    if df.empty or len(df) <= days:
        return df
    
    df_sorted = df.sort_values('date')
    return df_sorted.tail(days)

def prepare_features(df):
    """Подготавливает признаки для модели."""
    if df.empty:
        return df
    
    df = df.rename(columns={'total_sales': 'total_daily_sales'})
    
    for lag in [1, 7, 30]:
        df[f'sales_lag_{lag}'] = df['total_daily_sales'].shift(lag)
    
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    df = df.dropna()
    logger.info(f"После подготовки признаков осталось {len(df)} записей")
    
    return df

def train_model(session):
    """Обучает модель машинного обучения."""
    logger.info("Загрузка данных для обучения модели...")
    
    df = load_processed_data(session)
    
    # Используем все данные для обучения, но ограничиваем отображение
    if len(df) < 30:
        logger.warning(f"Мало данных для обучения: всего {len(df)} записей. Нужно минимум 30.")
        return None, None
    
    df = prepare_features(df)
    
    if df.empty:
        logger.warning("Нет данных после подготовки признаков")
        return None, None
    
    X = df[['day_of_week', 'month', 'day_of_month', 'is_weekend', 'sales_lag_1', 'sales_lag_7', 'sales_lag_30']]
    y = df['total_daily_sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    
    logger.info("Обучение модели RandomForest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    logger.info(f"Модель обучена!")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"MAPE: {mape:.2%}")
    logger.info(f"Обучено на {len(X_train)} записях")
    logger.info(f"Протестировано на {len(X_test)} записях")
    
    # Сохраняем точность в сессию
    accuracy_data = session.get('model_accuracy', [])
    accuracy_data.append({
        'accuracy': float(mape),
        'model_name': 'RandomForest',
        'created_at': datetime.now().isoformat()
    })
    session['model_accuracy'] = accuracy_data
    session.modified = True
    
    logger.info(f"Точность модели сохранена: {mape:.2%}")
    
    return model, mape

def make_predictions(model, session, days_to_forecast=7):
    """Делает прогноз на будущие даты."""
    if model is None:
        logger.error("Модель не обучена!")
        return None
    
    df = load_processed_data(session)
    
    if df.empty:
        logger.error("Нет данных для создания прогноза")
        return None
    
    df = df.rename(columns={'total_sales': 'total_daily_sales'})
    df_prepared = prepare_features(df)
    
    if df_prepared.empty:
        logger.error("Недостаточно данных для создания прогноза")
        return None
    
    last_known = df_prepared.iloc[-1].copy()
    predictions = []
    
    # Получаем последнюю дату из данных
    last_date = df_prepared['date'].iloc[-1]
    current_date = last_date + pd.Timedelta(days=1)  # Начинаем со следующего дня
    
    historical_data = df_prepared.set_index('date')['total_daily_sales']
    
    for i in range(days_to_forecast):
        lag_1_date = current_date - pd.Timedelta(days=1)
        lag_7_date = current_date - pd.Timedelta(days=7)
        lag_30_date = current_date - pd.Timedelta(days=30)
        
        sales_lag_1 = historical_data.get(lag_1_date, last_known['total_daily_sales'])
        sales_lag_7 = historical_data.get(lag_7_date, last_known['total_daily_sales'])
        sales_lag_30 = historical_data.get(lag_30_date, last_known['total_daily_sales'])
        
        features = pd.DataFrame([{
            'day_of_week': current_date.dayofweek,
            'month': current_date.month,
            'day_of_month': current_date.day,
            'is_weekend': 1 if current_date.dayofweek in [5, 6] else 0,
            'sales_lag_1': float(sales_lag_1),
            'sales_lag_7': float(sales_lag_7),
            'sales_lag_30': float(sales_lag_30)
        }])
        
        prediction = float(model.predict(features)[0])
        
        predictions.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'predicted_sales': prediction
        })
        
        current_date += pd.Timedelta(days=1)
    
    logger.info(f"Создано {len(predictions)} прогнозов")
    return predictions

def get_display_data(session, historical_days=30, forecast_days=7):
    """Возвращает данные для отображения на графике (ограниченные 30+7 дней)"""
    processed_data = session.get('processed_data', [])
    forecast_results = session.get('forecast_results', [])
    
    # Ограничиваем исторические данные
    if processed_data:
        df_hist = pd.DataFrame(processed_data)
        if 'date' in df_hist.columns:
            df_hist['date'] = pd.to_datetime(df_hist['date'])
            df_hist = df_hist.sort_values('date')
            # Берем только последние N дней
            limited_historical = df_hist.tail(historical_days)
            historical_display = limited_historical.to_dict('records')
        else:
            historical_display = processed_data[-historical_days:] if len(processed_data) > historical_days else processed_data
    else:
        historical_display = []
    
    # Ограничиваем прогноз если нужно
    if forecast_results and len(forecast_results) > forecast_days:
        forecast_display = forecast_results[:forecast_days]
    else:
        forecast_display = forecast_results
    
    return historical_display, forecast_display

if __name__ == "__main__":
    logger.info("Запуск ML-модуля прогнозирования...")
    
    # Для тестирования без сессии
    class MockSession:
        def __init__(self):
            self.data = {
                'processed_data': [],
                'model_accuracy': []
            }
        
        def get(self, key, default=None):
            return self.data.get(key, default)
        
        def __setitem__(self, key, value):
            self.data[key] = value
    
    mock_session = MockSession()
    model, accuracy = train_model(mock_session)
    
    if model is not None:
        predictions = make_predictions(model, mock_session, days_to_forecast=7)
        
        if predictions:
            logger.info("\nПрогноз продаж на следующие 7 дней:")
            for pred in predictions:
                logger.info(f"   {pred['date']}: {pred['predicted_sales']:.2f} руб.")