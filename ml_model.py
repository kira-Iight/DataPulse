# utils/ml_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime
import os
import sys

# Добавьте путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import get_engine
from sqlalchemy import text, insert
from models.models import forecast_results_table, model_accuracy_table
from loguru import logger

def load_processed_data():
    """Загружает обработанные данные из БД."""
    engine = get_engine()
    query = text("SELECT date, total_sales, day_of_week, month, is_holiday FROM processed_data ORDER BY date;")
    df = pd.read_sql(query, engine, parse_dates=['date'])
    return df

def prepare_features(df):
    """Подготавливает признаки для модели."""
    df = df.rename(columns={'total_sales': 'total_daily_sales'})
    
    for lag in [1, 7, 30]:
        df[f'sales_lag_{lag}'] = df['total_daily_sales'].shift(lag)
    
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    df = df.dropna()
    logger.info(f"После подготовки признаков осталось {len(df)} записей")
    
    return df

def train_model():
    """Обучает модель машинного обучения."""
    logger.info("Загрузка данных для обучения модели...")
    
    df = load_processed_data()
    
    if len(df) < 30:
        logger.warning(f"Мало данных для обучения: всего {len(df)} записей. Нужно минимум 30.")
        return None, None
    
    df = prepare_features(df)
    
    X = df[['day_of_week', 'month', 'day_of_month', 'is_weekend', 'sales_lag_1', 'sales_lag_7', 'sales_lag_30']]
    y = df['total_daily_sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    
    logger.info("Обучение модели RandomForest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    logger.info(f"✅ Модель обучена!")
    logger.info(f"   MAE: {mae:.2f}")
    logger.info(f"   MAPE: {mape:.2%}")
    logger.info(f"   Обучено на {len(X_train)} записях")
    logger.info(f"   Протестировано на {len(X_test)} записях")
    
    # Сохраняем точность в БД
    try:
        engine = get_engine()
        with engine.connect() as conn:
            stmt = insert(model_accuracy_table).values(
                accuracy=mape,
                model_name='RandomForest',
                created_at=datetime.now()
            )
            conn.execute(stmt)
            conn.commit()
            logger.info(f"Точность модели сохранена в БД: {mape:.2%}")
    except Exception as e:
        logger.error(f"Ошибка сохранения точности модели: {e}")
    
    return model, mape

def save_predictions_to_db(predictions):
    """Сохраняет прогнозы в базу данных."""
    if not predictions:
        logger.warning("Нет прогнозов для сохранения")
        return False
    
    engine = get_engine()
    
    try:
        # Очищаем старые прогнозы
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM forecast_results"))
            conn.commit()
        
        # Сохраняем новые прогнозы
        with engine.connect() as conn:
            for pred in predictions:
                # Преобразуем numpy типы в стандартные Python типы
                predicted_amount = float(pred['predicted_sales'])  # Преобразуем np.float64 в float
                target_date = pred['date'].date() if hasattr(pred['date'], 'date') else pred['date']
                
                stmt = insert(forecast_results_table).values(
                    forecast_date=datetime.now().date(),
                    target_date=target_date,
                    predicted_amount=predicted_amount,
                    model_name='RandomForest',
                    confidence_interval='±20%'
                )
                conn.execute(stmt)
            conn.commit()
        
        logger.info(f"✅ Сохранено {len(predictions)} прогнозов в базу данных")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении прогнозов в БД: {e}")
        return False

def make_predictions(model, days_to_forecast=7):
    """Делает прогноз на будущие даты и сохраняет в БД."""
    if model is None:
        logger.error("Модель не обучена!")
        return None
    
    engine = get_engine()
    
    # Проверим, есть ли данные в processed_data
    check_query = "SELECT COUNT(*) as count FROM processed_data;"
    count_result = pd.read_sql(check_query, engine)
    logger.info(f"Записей в processed_data: {count_result['count'].iloc[0]}")
    
    query = text("""
        SELECT date, total_sales, day_of_week, month 
        FROM processed_data 
        ORDER BY date DESC 
        LIMIT 60
    """)
    recent_data = pd.read_sql(query, engine, parse_dates=['date'])
    recent_data = recent_data.sort_values('date')
    
    recent_data = recent_data.rename(columns={'total_sales': 'total_daily_sales'})
    df = prepare_features(recent_data)
    
    if len(df) == 0:
        logger.error("Недостаточно данных для создания прогноза")
        return None
    
    last_known = df.iloc[-1].copy()
    predictions = []
    
    # Получаем последнюю дату из данных
    if not df.empty:
        last_date = df['date'].iloc[-1]
        current_date = last_date + pd.Timedelta(days=1)  # Начинаем со следующего дня после последней известной даты
    else:
        current_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    
    historical_data = recent_data.set_index('date')['total_daily_sales']
    
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
            'sales_lag_1': float(sales_lag_1),  # Преобразуем в float
            'sales_lag_7': float(sales_lag_7),  # Преобразуем в float
            'sales_lag_30': float(sales_lag_30)  # Преобразуем в float
        }])
        
        prediction = float(model.predict(features)[0])  # Преобразуем в float
        
        predictions.append({
            'date': current_date,
            'predicted_sales': prediction
        })
        
        current_date += pd.Timedelta(days=1)
    
    # Сохраняем прогнозы в базу данных
    save_predictions_to_db(predictions)
    
    return predictions

if __name__ == "__main__":
    logger.info("🚀 Запуск ML-модуля прогнозирования...")
    
    model, accuracy = train_model()
    
    if model is not None:
        predictions = make_predictions(model, days_to_forecast=7)
        
        if predictions:
            logger.info("\n📈 Прогноз продаж на следующие 7 дней:")
            for pred in predictions:
                logger.info(f"   {pred['date'].strftime('%Y-%m-%d')}: {pred['predicted_sales']:.2f} руб.")