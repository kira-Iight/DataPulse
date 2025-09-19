import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Теперь импортируем metadata
from models.models import metadata  # Правильный импорт

load_dotenv()


DB_URL = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

engine = create_engine(DB_URL)

# ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ВСЕХ ТАБЛИЦ
def create_tables():
    """Создает все таблицы в базе данных на основе их описания."""
    try:
        metadata.create_all(engine)
        print("Таблицы успешно созданы в базе данных!")
        
        # Проверим, какие таблицы действительно создались
        with engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
            tables = [row[0] for row in result]
            print(f"Созданные таблицы: {tables}")
            
    except Exception as e:
        print(f"Произошла ошибка при создании таблиц: {e}")

# Функция для проверки подключения
def test_connection():
    """Проверяет подключение к базе данных."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.scalar()
            print("Подключение к PostgreSQL успешно!")
            print(f"Версия PostgreSQL: {version}")
            return True
    except Exception as e:
        print(f"Ошибка подключения к PostgreSQL: {e}")
        return False

def get_engine():
    """Возвращает объект engine для использования в других модулях."""
    return engine

if __name__ == "__main__":
    if test_connection():
        create_tables()