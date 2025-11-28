# logging_config.py
import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(logs_dir="logs", app_name="sales_forecast"):
    """Настраивает систему логирования с обработкой ошибок"""
    
    try:
        # Создаем директорию для логов
        os.makedirs(logs_dir, exist_ok=True)
        
        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Файловый обработчик
        log_file = os.path.join(logs_dir, f'{app_name}.log')
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Настраиваем корневой логгер
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Удаляем существующие обработчики
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Устанавливаем уровень для внешних библиотек
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('weasyprint').setLevel(logging.WARNING)
        
        logging.info("Система логирования инициализирована")
        return True
        
    except Exception as e:
        # Fallback: базовая настройка
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.error(f"Не удалось настроить файловое логирование: {e}")
        return False