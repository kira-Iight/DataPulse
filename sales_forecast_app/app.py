# app.py
import os
import sys
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
from loguru import logger

# Настройка логирования
logger.remove()
logger.add("app.log", rotation="500 MB", level="INFO")

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'

# Конфигурация загрузки файлов
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Убедимся, что папка для загрузок существует
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_session_data():
    """Инициализирует структуры данных в сессии"""
    if 'raw_data' not in session:
        session['raw_data'] = []
    if 'processed_data' not in session:
        session['processed_data'] = []
    if 'forecast_results' not in session:
        session['forecast_results'] = []
    if 'model_accuracy' not in session:
        session['model_accuracy'] = []
    session.modified = True

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

def save_processed_data(df):
    """Сохраняет обработанные данные в сессию."""
    init_session_data()
    # Конвертируем DataFrame в список словарей
    processed_data = df.to_dict('records')
    # Преобразуем даты в строки для сериализации
    for item in processed_data:
        if hasattr(item['date'], 'strftime'):
            item['date'] = item['date'].strftime('%Y-%m-%d')
    session['processed_data'] = processed_data
    session.modified = True

@app.route('/')
def index():
    """Главная страница - отдает index.html"""
    init_session_data()
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """API endpoint для получения статистики"""
    try:
        init_session_data()
        processed_data = session.get('processed_data', [])
        # Количество записей
        total_records = len(processed_data)
        # Средние продажи
        avg_sales = 0
        if total_records > 0:
            total_sales = sum(item.get('total_sales', 0) for item in processed_data)
            avg_sales = total_sales / total_records
        # Точность прогноза (последний расчет)
        accuracy_data = session.get('model_accuracy', [])
        accuracy = accuracy_data[-1]['accuracy'] if accuracy_data else None
        return jsonify({
            'total_records': int(total_records),
            'avg_sales': round(float(avg_sales), 2),
            'accuracy': accuracy
        }), 200
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        return jsonify({'error': str(e)}), 500

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
            # Сохраняем сырые данные в сессию
            init_session_data()
            session['raw_data'] = df_raw.to_dict('records')
            df_processed = preprocess_data(df_raw)
            logger.info(f"Обработано {len(df_processed)} строк")
            save_processed_data(df_processed)
            logger.info("Данные успешно сохранены в сессии")
            # Удаляем временный файл
            os.remove(filepath)
            return jsonify({'message': 'Файл успешно загружен и обработан', 'rows': len(df_processed)}), 200
        except Exception as e:
            logger.error(f"Ошибка при обработке файла: {e}")
            return jsonify({'error': f'Ошибка обработки файла: {str(e)}'}), 500
    else:
        logger.warning("Недопустимый тип файла")
        return jsonify({'error': 'Недопустимый тип файла. Разрешен только .csv'}), 400

@app.route('/api/data/raw')
def get_raw_data():
    """API endpoint для получения сырых данных из сессии"""
    logger.info("Запрос сырых данных")
    try:
        init_session_data()
        raw_data = session.get('raw_data', [])
        return jsonify(raw_data[:100]), 200
    except Exception as e:
        logger.error(f"Ошибка получения сырых данных: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/processed')
def get_processed_data():
    """API endpoint для получения обработанных данных из сессии"""
    logger.info("Запрос обработанных данных")
    try:
        init_session_data()
        processed_data = session.get('processed_data', [])
        return jsonify(processed_data[:100]), 200
    except Exception as e:
        logger.error(f"Ошибка получения обработанных данных: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def run_forecast():
    """API endpoint для запуска прогнозирования"""
    logger.info("Запрос на запуск прогнозирования")
    try:
        from utils.ml_model import train_model, make_predictions
        # Обучаем модель
        model, accuracy = train_model(session)
        if model is None:
            logger.error("Модель не обучена")
            return jsonify({'error': 'Недостаточно данных для обучения модели'}), 400
        # Делаем прогноз на 7 дней вперед
        predictions = make_predictions(model, session, days_to_forecast=7)
        if predictions:
            logger.info("Прогноз успешно выполнен")
            # Сохраняем прогноз в сессию
            session['forecast_results'] = predictions
            session.modified = True
            # Форматируем прогноз для отправки
            formatted_predictions = [
                {
                    'date': pred['date'].strftime('%Y-%m-%d') if hasattr(pred['date'], 'strftime') else pred['date'],
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
        from utils.report_generator import generate_report
        
        pdf_buffer = generate_report(session)
        
        if pdf_buffer:
            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name='sales_forecast_report.pdf',
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': 'Не удалось сгенерировать отчет'}), 500
            
    except Exception as e:
        logger.error(f"Ошибка при генерации отчета: {e}")
        return jsonify({'error': str(e)}), 500

# app.py - добавим новые маршруты

@app.route('/api/report/sales')
def generate_sales_report():
    """API endpoint для генерации отчета только по историческим данным"""
    logger.info("Запрос на генерацию отчета по продажам")
    try:
        from utils.report_generator import generate_sales_report
        
        pdf_buffer = generate_sales_report(session)
        
        if pdf_buffer:
            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name='sales_report.pdf',
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': 'Нет данных для генерации отчета по продажам'}), 400
            
    except Exception as e:
        logger.error(f"Ошибка при генерации отчета по продажам: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/report/forecast')
def generate_forecast_report():
    """API endpoint для генерации отчета только по прогнозам"""
    logger.info("Запрос на генерацию отчета по прогнозам")
    try:
        from utils.report_generator import generate_forecast_report
        
        pdf_buffer = generate_forecast_report(session)
        
        if pdf_buffer:
            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name='forecast_report.pdf',
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': 'Нет данных прогноза для генерации отчета'}), 400
            
    except Exception as e:
        logger.error(f"Ошибка при генерации отчета по прогнозам: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/report/full')
def generate_full_report():
    """API endpoint для генерации полного отчета"""
    logger.info("Запрос на генерацию полного отчета")
    try:
        from utils.report_generator import generate_full_report
        
        pdf_buffer = generate_full_report(session)
        
        if pdf_buffer:
            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name='full_sales_report.pdf',
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': 'Не удалось сгенерировать полный отчет'}), 400
            
    except Exception as e:
        logger.error(f"Ошибка при генерации полного отчета: {e}")
        return jsonify({'error': str(e)}), 500

# Вспомогательная функция для получения названия дня недели
def get_day_name(day_number):
    days = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    return days[day_number] if 0 <= day_number < 7 else 'Неизвестно'

@app.route('/api/data/clear')
def clear_data():
    """API endpoint для очистки данных из сессии"""
    logger.info("Запрос на очистку данных")
    try:
        session.clear()
        init_session_data()
        logger.info("Данные успешно очищены")
        return jsonify({'message': 'Данные успешно очищены'}), 200
    except Exception as e:
        logger.error(f"Ошибка при очистке данных: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Запуск Flask-приложения...")
    app.run(debug=True, host='0.0.0.0', port=5500)