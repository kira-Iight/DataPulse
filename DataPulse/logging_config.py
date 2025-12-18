import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(logs_dir="logs", app_name="sales_forecast"):
    """Настройка системы логирования для приложения DataPulse.
    Создает гибкую систему логирования с поддержкой:
    - Ротации лог-файлов (архивирование старых файлов)
    - Одновременной записи в файл и вывод в консоль
    - Контроля уровня логирования для различных модулей"""
    
    try:
        # Создание директории для логов (если не существует)
        os.makedirs(logs_dir, exist_ok=True)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'  # Формат даты: ГГГГ-ММ-ДД ЧЧ:ММ:СС
        )
        
        # Настройка файлового обработчика с ротацией
        # Файл будет автоматически архивироваться при достижении 5 МБ
        log_file = os.path.join(logs_dir, f'{app_name}.log')
        file_handler = RotatingFileHandler(
            filename=log_file,           # Путь к основному лог-файлу
            maxBytes=5 * 1024 * 1024,   # Максимальный размер файла (5 МБ)
            backupCount=3,               # Количество архивных копий (сохраняет 3 старых файла)
            encoding='utf-8'             # Кодировка файла для поддержки кириллицы
        )
        file_handler.setFormatter(formatter)  # Применение формата сообщений
        file_handler.setLevel(logging.INFO)   # Уровень логирования для файла
        
        # Настройка консольного обработчика
        console_handler = logging.StreamHandler()  # Вывод в стандартный поток (консоль)
        console_handler.setFormatter(formatter)    # Применение того же формата сообщений
        console_handler.setLevel(logging.INFO)     # Уровень логирования для консоли
        
        # Настройка корневого логгера (базового для всего приложения)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Установка минимального уровня логирования
        
        # Удаление существующих обработчиков для избежания дублирования
        # Это важно при повторном вызове функции настройки
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Добавление обработчиков к корневому логгеру
        root_logger.addHandler(file_handler)    # Обработчик для записи в файл
        root_logger.addHandler(console_handler) # Обработчик для вывода в консоль
        
        # Настройка уровней логирования для внешних библиотек
        # Уменьшение шума от библиотек, которые могут генерировать много сообщений
        logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Библиотека графиков
        logging.getLogger('PIL').setLevel(logging.WARNING)         # Библиотека обработки изображений
        logging.getLogger('weasyprint').setLevel(logging.WARNING)  # Библиотека генерации PDF
        
        # Логирование успешной инициализации системы
        logging.info("Система логирования инициализирована")
        return True  # Успешное завершение настройки
        
    except Exception as e:
        # Обработка ошибок при настройке системы логирования
        # Используем базовую конфигурацию logging как запасной вариант
        
        # Базовая настройка logging (работает даже если файловая система недоступна)
        logging.basicConfig(
            level=logging.INFO,  # Уровень логирования
            format='%(asctime)s - %(levelname)s - %(message)s',  # Простой формат
            datefmt='%Y-%m-%d %H:%M:%S'  # Формат даты
        )
        
        # Логирование ошибки настройки
        logging.error(f"Не удалось настроить файловое логирование: {e}")
        return False  # Настройка не удалась, но приложение продолжит работать