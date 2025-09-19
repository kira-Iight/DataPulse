# app.py
import os
import sys

# Добавьте текущую директорию в путь Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import logging

# Абсолютные импорты
from utils.database import test_connection, create_tables, get_engine
from utils.data_processor import load_data_from_csv, preprocess_data, save_processed_data
from utils.ml_model import train_model, make_predictions
from utils.report_generator import generate_report
from utils.create_extended_data import create_extended_test_data, load_and_process_extended_data
from loguru import logger

# Настройка логирования
logger.remove()
logger.add("app.log", rotation="500 MB", level="INFO")

app = Flask(__name__)
app.secret_key = 'supersecretkey' # Замените на более безопасный ключ в production

# Конфигурация загрузки файлов
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Убедимся, что папка для загрузок существует
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Главная страница - отдает index.html"""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """API endpoint для получения статистики"""
    try:
        engine = get_engine()
        
        # Количество записей
        count_query = "SELECT COUNT(*) as count FROM processed_data;"
        count_result = pd.read_sql(count_query, engine)
        total_records = count_result['count'].iloc[0]
        
        # Средние продажи
        avg_query = "SELECT AVG(total_sales) as avg_sales FROM processed_data;"
        avg_result = pd.read_sql(avg_query, engine)
        avg_sales = avg_result['avg_sales'].iloc[0] or 0
        
        # Точность прогноза (последний расчет)
        accuracy_query = "SELECT accuracy FROM model_accuracy ORDER BY created_at DESC LIMIT 1;"
        try:
            accuracy_result = pd.read_sql(accuracy_query, engine)
            accuracy = accuracy_result['accuracy'].iloc[0] if not accuracy_result.empty else None
        except:
            accuracy = None
        
        return jsonify({
            'total_records': int(total_records),
            'avg_sales': round(float(avg_sales), 2),
            'accuracy': accuracy
        }), 200
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/activity')
def get_activity():
    """API endpoint для получения лога действий"""
    # В реальном приложении это бы бралось из БД
    activities = [
        {
            'action': 'Система запущена',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'Успешно'
        }
    ]
    return jsonify(activities), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """API endpoint для загрузки CSV файла"""
    logger.info("Получен запрос на загрузку файла")
    if 'file' not in request.files:
        logger.warning("Файл не найден в запросе")
        return jsonify({'error': 'Файл не выбран'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("Имя файла пустое")
        return jsonify({'error': 'Файл не выбран'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Добавим временную метку к имени файла для уникальности
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        logger.info(f"Файл сохранен: {filepath}")
        
        try:
            # Обрабатываем данные
            df_raw = load_data_from_csv(filepath)
            logger.info(f"Загружено {len(df_raw)} строк сырых данных")
            df_processed = preprocess_data(df_raw)
            logger.info(f"Обработано {len(df_processed)} строк")
            save_processed_data(df_processed)
            logger.info("Данные успешно сохранены в БД")
            return jsonify({'message': 'Файл успешно загружен и обработан', 'rows': len(df_processed)}), 200
        except Exception as e:
            logger.error(f"Ошибка при обработке файла: {e}")
            return jsonify({'error': f'Ошибка обработки файла: {str(e)}'}), 500
    else:
        logger.warning("Недопустимый тип файла")
        return jsonify({'error': 'Недопустимый тип файла. Разрешен только .csv'}), 400

@app.route('/api/data/raw')
def get_raw_data():
    """API endpoint для получения сырых данных из БД"""
    logger.info("Запрос сырых данных")
    try:
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM raw_sales_data ORDER BY date DESC LIMIT 100", engine)
        # Преобразуем дату в строку для сериализации
        df['date'] = df['date'].astype(str)
        return jsonify(df.to_dict(orient='records')), 200
    except Exception as e:
        logger.error(f"Ошибка получения сырых данных: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/processed')
def get_processed_data():
    """API endpoint для получения обработанных данных из БД"""
    logger.info("Запрос обработанных данных")
    try:
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM processed_data ORDER BY date DESC LIMIT 100", engine)
        # Преобразуем дату в строку для сериализации
        df['date'] = df['date'].astype(str)
        return jsonify(df.to_dict(orient='records')), 200
    except Exception as e:
        logger.error(f"Ошибка получения обработанных данных: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def run_forecast():
    """API endpoint для запуска прогнозирования"""
    logger.info("Запрос на запуск прогнозирования")
    try:
        # Обучаем модель
        model, accuracy = train_model()
        if model is None:
            logger.error("Модель не обучена")
            return jsonify({'error': 'Недостаточно данных для обучения модели'}), 400
        
        # Делаем прогноз на 7 дней вперед
        predictions = make_predictions(model, days_to_forecast=7)
        if predictions:
            logger.info("Прогноз успешно выполнен")
            # Форматируем прогноз для отправки
            formatted_predictions = [
                {
                    'date': pred['date'].strftime('%Y-%m-%d'),
                    'predicted_sales': round(pred['predicted_sales'], 2)
                } for pred in predictions
            ]
            return jsonify({
                'message': 'Прогноз успешно выполнен',
                'accuracy': accuracy,
                'predictions': formatted_predictions
            }), 200
        else:
            logger.error("Прогноз не был сгенерирован")
            return jsonify({'error': 'Не удалось сгенерировать прогноз'}), 500
    except Exception as e:
        logger.error(f"Ошибка при выполнении прогноза: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/report/generate')
def generate_pdf_report():
    """API endpoint для генерации PDF отчета"""
    logger.info("Запрос на генерацию PDF отчета")
    try:
        generate_report()
        logger.info("PDF отчет сгенерирован")
        return send_file('sales_forecast_report.pdf', as_attachment=True)
    except Exception as e:
        logger.error(f"Ошибка при генерации отчета: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/clear')
def clear_data():
    """API endpoint для очистки данных из БД"""
    logger.info("Запрос на очистку данных")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute("DELETE FROM raw_sales_data")
            conn.execute("DELETE FROM processed_data")
            conn.execute("DELETE FROM forecast_results")
            conn.commit()
        logger.info("Данные успешно очищены")
        return jsonify({'message': 'Данные успешно очищены'}), 200
    except Exception as e:
        logger.error(f"Ошибка при очистке данных: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/create_extended')
def create_extended_data_route():
    """API endpoint для создания и загрузки расширенных тестовых данных"""
    logger.info("Запрос на создание расширенных тестовых данных")
    try:
        # Создаем расширенные данные
        create_extended_test_data()
        # Загружаем и обрабатываем их
        load_and_process_extended_data()
        logger.info("Расширенные тестовые данные созданы и загружены")
        return jsonify({'message': 'Расширенные тестовые данные успешно созданы и загружены'}), 200
    except Exception as e:
        logger.error(f"Ошибка при создании расширенных данных: {e}")
        return jsonify({'error': str(e)}), 500



@app.before_request
def before_first_request():
    """Инициализация при первом запросе"""
    if not hasattr(app, 'initialized'):
        logger.info("Инициализация приложения...")
        if test_connection():
            logger.info("Подключение к БД установлено")
            create_tables()
        else:
            logger.error("Не удалось подключиться к БД")
        app.initialized = True

if __name__ == '__main__':
    # Выполняем инициализацию при запуске
    logger.info("Инициализация приложения...")
    if test_connection():
        logger.info("Подключение к БД установлено")
        create_tables()
    else:
        logger.error("Не удалось подключиться к БД")
    
    logger.info("Запуск Flask-приложения...")
    app.run(debug=True, host='0.0.0.0', port=5500)