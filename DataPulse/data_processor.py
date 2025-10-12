#data_processor.py
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data_from_csv(file_path):
    """Загружает данные из CSV файла."""
    df = pd.read_csv(file_path, parse_dates=['date'])
    return df

def preprocess_data(df):
    """Очищает и агрегирует данные."""
    # Удаление дубликатов, обработка пропусков
    df_clean = df.drop_duplicates().dropna()
    
    # Агрегация по дням
    df_daily = df_clean.groupby('date', as_index=False).agg({'quantity': 'sum', 'price': 'mean'})
    df_daily['total_sales'] = df_daily['quantity'] * df_daily['price']
    
    # Создание признаков
    df_daily['day_of_week'] = df_daily['date'].dt.dayofweek
    df_daily['month'] = df_daily['date'].dt.month
    df_daily['is_holiday'] = False
    
    return df_daily[['date', 'total_sales', 'day_of_week', 'month', 'is_holiday']]


if __name__ == "__main__":
    # Пример использования
    df_raw = load_data_from_csv('/Users/nayeon/Documents/5S_coding/sales_forecast_app/data/sales_data.csv')
    print("Данные загружены из CSV:")
    print(df_raw.head())
    
    df_processed = preprocess_data(df_raw)
    print("\nДанные после обработки:")
    print(df_processed.head())