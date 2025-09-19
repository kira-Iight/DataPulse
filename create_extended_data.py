# utils/create_extended_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
#from .data_processor import preprocess_data, save_processed_data # Импорт внутри функции, чтобы избежать циклических импортов
import os
# Добавьте путь к корневой директории проекта
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Получаем путь к директории текущего скрипта
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Определяем путь к папке data на уровень выше
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'data')
# Убедимся, что папка существует
os.makedirs(DATA_DIR, exist_ok=True)

def create_extended_test_data():
    """Создает расширенные тестовые данные на 90 дней."""
    # Генерируем даты со случайным смещением
    random_offset = np.random.randint(0, 365)
    start_date = datetime(2024, 1, 1) + timedelta(days=random_offset)
    dates = [start_date + timedelta(days=i) for i in range(90)]

    print(f"Генерируем данные с {start_date.date()} на 90 дней")
    
    # Базовые продукты
    products = ['Product A', 'Product B', 'Product C']
    base_prices = {'Product A': 100.50, 'Product B': 200.00, 'Product C': 150.00}
    
    # Создаем данные с сезонностью и трендом
    data = []
    for date in dates:
        is_weekend = date.weekday() in [5, 6]
        month_factor = 1 + (date.month - 1) * 0.02
        weekend_factor = 1.2 if is_weekend else 1.0
        
        for product in products:
            base_quantity = np.random.randint(5, 15)
            quantity = int(base_quantity * month_factor * weekend_factor * np.random.uniform(0.9, 1.1))
            price = base_prices[product] * np.random.uniform(0.95, 1.05)
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'product_name': product,
                'quantity': max(1, quantity),
                'price': round(price, 2)
            })
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    # Сохраняем в обычный файл (не временный)
    csv_path = os.path.join(DATA_DIR, 'extended_sales_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Создан файл с данными от {df['date'].min()} до {df['date'].max()}: {csv_path}")
    return df

def load_and_process_extended_data():
    """Загружает и обрабатывает расширенные данные."""
    from utils.data_processor import preprocess_data
    from utils.database import get_engine
    
    # Загружаем расширенные данные
    csv_path = os.path.join(DATA_DIR, 'extended_sales_data.csv')
    
    # Проверяем, что файл существует
    if not os.path.exists(csv_path):
        print(f"Файл {csv_path} не существует!")
        return
    
    # Очищаем кэш pandas
    if hasattr(pd.io.parsers, '_parser_cache'):
        parser_cache = pd.io.parsers._parser_cache
        for key in list(parser_cache.keys()):
            if csv_path in str(key):
                del parser_cache[key]
    
    # Читаем файл с нуля
    try:
        df_raw = pd.read_csv(csv_path, parse_dates=['date'])
        print(f"Загружено {len(df_raw)} строк сырых данных")
        print(f"Диапазон дат: от {df_raw['date'].min()} до {df_raw['date'].max()}")
        print("Первые 5 записей:")
        print(df_raw.head())
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return
    
    # Обрабатываем данные
    df_processed = preprocess_data(df_raw)
    print(f"\nПосле обработки: {len(df_processed)} записей")
    print("Первые 5 обработанных записей:")
    print(df_processed.head())
    
    # Сохраняем в БД
    engine = get_engine()
    
    # Сначала удаляем данные с такими же датами
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            # Удаляем данные с датами, которые есть в новых данных
            dates_to_delete = df_processed['date'].unique()
            for date in dates_to_delete:
                conn.execute(text(f"DELETE FROM processed_data WHERE date = '{date}'"))
            conn.commit()
        
        # Добавляем новые данные
        df_processed.to_sql('processed_data', engine, if_exists='append', index=False)
        print("\nДанные добавлены в базу данных!")
        
        # Проверяем общее количество записей в БД
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM processed_data"))
            total_count = result.scalar()
            print(f"Всего записей в БД: {total_count}")
            
    except Exception as e:
        print(f"Ошибка при работе с БД: {e}")

if __name__ == "__main__":
    # Создаем расширенные данные
    create_extended_test_data()
    # Загружаем и обрабатываем их
    load_and_process_extended_data()

def load_and_process_extended_data():
    """Загружает и обрабатывает расширенные данные."""
    from utils.data_processor import preprocess_data
    from utils.database import get_engine
    
    # Загружаем расширенные данные - ПРАВИЛЬНОЕ ЧТЕНИЕ
    csv_path = os.path.join(DATA_DIR, 'extended_sales_data.csv')
    
    # Полностью очищаем кэш pandas для этого файла
    if hasattr(pd.io.parsers, '_parser_cache'):
        if csv_path in pd.io.parsers._parser_cache:
            del pd.io.parsers._parser_cache[csv_path]
    
    # Читаем файл с нуля
    df_raw = pd.read_csv(csv_path, parse_dates=['date'])
    print(f"Загружено {len(df_raw)} строк сырых данных")
    print("Первые 5 записей:")
    print(df_raw.head())
    print(f"Диапазон дат: от {df_raw['date'].min()} до {df_raw['date'].max()}")
    
    # Обрабатываем данные
    df_processed = preprocess_data(df_raw)
    print(f"\nПосле обработки: {len(df_processed)} записей")
    print("Первые 5 обработанных записей:")
    print(df_processed.head())
    
    # Сохраняем в БД с ДОБАВЛЕНИЕМ данных вместо замены
    engine = get_engine()
    
    # Сначала удаляем данные с такими же датами (если они есть)
    from sqlalchemy import text
    with engine.connect() as conn:
        # Удаляем только те данные, даты которых есть в новых данных
        dates_to_delete = df_processed['date'].unique()
        for date in dates_to_delete:
            conn.execute(text(f"DELETE FROM processed_data WHERE date = '{date}'"))
        conn.commit()
    
    # ДОБАВЛЯЕМ новые данные (не заменяем всю таблицу)
    df_processed.to_sql('processed_data', engine, if_exists='append', index=False)
    print("\nДанные добавлены в базу данных!")
    # Удаляем временный файл
    os.unlink(csv_path)
    # Проверяем общее количество записей в БД
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM processed_data"))
        total_count = result.scalar()
        print(f"Всего записей в БД: {total_count}")

if __name__ == "__main__":
    # Создаем расширенные данные
    create_extended_test_data()
    # Загружаем и обрабатываем их
    load_and_process_extended_data()
