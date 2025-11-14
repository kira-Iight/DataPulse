# report_generator.py
import pandas as pd
import numpy as np
from jinja2 import Template
from weasyprint import HTML
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import logging
logger = logging.getLogger(__name__)

def generate_sales_report(session_data):
    """Генерирует отчет по историческим данным"""
    try:
        processed_data = session_data.get('processed_data', [])
        if not processed_data:
            logger.warning("Нет данных для отчета по продажам")
            return None
        df = pd.DataFrame(processed_data)
        
        # Базовая валидация данных
        if 'total_sales' not in df.columns or 'date' not in df.columns:
            logger.error("Отсутствуют необходимые колонки в данных")
            return None
        
        # Конвертируем даты
        df['date'] = pd.to_datetime(df['date'])
        
        # Вычисляем статистику по ВСЕМ данным (не ограничиваем)
        stats = _calculate_sales_statistics(df)
        
        # Создаем график по ВСЕМ данным
        plot_base64 = _create_sales_plot(df, stats)
        if not plot_base64:
            return None
        
        # Генерируем HTML с полными данными
        html_content = _render_sales_html(stats, processed_data, plot_base64)
        
        # Конвертируем в PDF
        return _html_to_pdf(html_content)
        
    except Exception as e:
        logger.error(f"Ошибка генерации отчета по продажам: {e}")
        return None

def generate_forecast_report(session_data):
    """Генерирует отчет по прогнозам"""
    try:
        forecast_results = session_data.get('forecast_results', [])
        processed_data = session_data.get('processed_data', [])
        model_accuracy = session_data.get('model_accuracy', [])
        
        if not forecast_results:
            logger.warning("Нет данных прогноза для отчета")
            return None
        
        # Ограничиваем исторические данные последними 60 днями (вместо 30)
        limited_historical = _get_limited_historical_data(processed_data, days=60)
        
        # Получаем информацию о модели
        model_info = _get_model_info(model_accuracy)
        
        # Создаем график
        plot_base64 = _create_forecast_plot(limited_historical, forecast_results, model_info)
        
        # Генерируем HTML
        html_content = _render_forecast_html(forecast_results, model_info, plot_base64)
        
        return _html_to_pdf(html_content)
        
    except Exception as e:
        logger.error(f"Ошибка генерации отчета по прогнозам: {e}")
        return None

def generate_full_report(session_data):
    """Генерирует полный отчет"""
    try:
        processed_data = session_data.get('processed_data', [])
        forecast_results = session_data.get('forecast_results', [])
        model_accuracy = session_data.get('model_accuracy', [])
        
        if not processed_data:
            logger.warning("Нет данных для полного отчета")
            return None
        
        # Ограничиваем исторические данные последними 60 днями
        limited_historical = _get_limited_historical_data(processed_data, days=60)
        
        # Статистика по полным историческим данным
        full_historical_stats = _calculate_sales_statistics(pd.DataFrame(processed_data))
        forecast_stats = _calculate_forecast_statistics(forecast_results)
        model_info = _get_model_info(model_accuracy)
        
        # График
        plot_base64 = _create_full_plot(limited_historical, forecast_results, 
                                      full_historical_stats, forecast_stats, model_info)
        
        # HTML - используем ограниченные исторические данные для отображения
        html_content = _render_full_html(limited_historical, forecast_results, 
                                       full_historical_stats, forecast_stats, model_info, plot_base64)
        
        return _html_to_pdf(html_content)
        
    except Exception as e:
        logger.error(f"Ошибка генерации полного отчета: {e}")
        return None

def _calculate_sales_statistics(df):
    """Вычисляет статистику продаж"""
    if df.empty or 'total_sales' not in df.columns:
        return {}
    
    sales = df['total_sales']
    
    # Дополнительные метрики
    total_days = len(sales)
    avg_growth = _calculate_average_growth(sales)
    
    # Исправляем обработку дат
    best_day_info = 'N/A'
    worst_day_info = 'N/A'
    
    if not sales.empty:
        best_day_idx = sales.idxmax()
        worst_day_idx = sales.idxmin()
        
        best_day_sales = float(sales.iloc[best_day_idx])
        worst_day_sales = float(sales.iloc[worst_day_idx])
        
        # Безопасное извлечение даты
        if 'date' in df.columns:
            best_day_date = df.iloc[best_day_idx]['date']
            worst_day_date = df.iloc[worst_day_idx]['date']
            
            best_day_info = _format_date_for_display(best_day_date)
            worst_day_info = _format_date_for_display(worst_day_date)
    else:
        best_day_sales = 0
        worst_day_sales = 0
    
    return {
        'total_sales': float(sales.sum()),
        'avg_daily': float(sales.mean()),
        'max_sales': float(sales.max()),
        'min_sales': float(sales.min()),
        'std_sales': float(sales.std()),
        'growth_rate': _calculate_growth_rate(sales),
        'total_days': total_days,
        'avg_growth': avg_growth,
        'best_day_date': best_day_info,
        'best_day_sales': best_day_sales,
        'worst_day_date': worst_day_info,
        'worst_day_sales': worst_day_sales
    }

def _calculate_average_growth(sales_data):
    """Вычисляет средний дневной рост"""
    if len(sales_data) < 2:
        return 0.0
    
    daily_growth = sales_data.pct_change().dropna()
    return float(daily_growth.mean() * 100)  # в процентах

def _calculate_forecast_statistics(forecast_results):
    """Вычисляет статистику прогноза"""
    if not forecast_results:
        return {}
    
    predictions = [f['predicted_sales'] for f in forecast_results]
    
    # Дополнительные метрики для прогноза
    total_growth = ((predictions[-1] - predictions[0]) / predictions[0] * 100) if predictions[0] > 0 else 0
    
    return {
        'total_forecast': sum(predictions),
        'avg_forecast': np.mean(predictions),
        'max_forecast': max(predictions),
        'min_forecast': min(predictions),
        'total_growth': total_growth,
        'days_count': len(predictions)
    }

def _calculate_growth_rate(sales_data):
    """Вычисляет общий темп роста"""
    if len(sales_data) < 2 or sales_data.iloc[0] == 0:
        return 0.0
    return ((sales_data.iloc[-1] - sales_data.iloc[0]) / sales_data.iloc[0]) * 100

def _get_limited_historical_data(processed_data, days=60):
    """Ограничивает исторические данные для графиков"""
    if not processed_data:
        return []
    
    df = pd.DataFrame(processed_data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').tail(days)
    return df.to_dict('records')

def _get_model_info(model_accuracy):
    """Извлекает информацию о модели"""
    if not model_accuracy:
        return {
            'name': 'Ridge Регрессия', 
            'date': datetime.datetime.now().strftime('%d.%m.%Y'),
            'mae': 0,
            'rmse': 0,
            'mae_absolute': 0,
            'rmse_absolute': 0,
            'features_used': 7,
            'training_size': 'N/A'
        }
    
    latest = model_accuracy[-1]
    return {
        'name': latest.get('model_name', 'Ridge Регрессия'),
        'date': latest.get('created_at', datetime.datetime.now().strftime('%d.%m.%Y'))[:10],
        'mae': latest.get('mae', 0),
        'rmse': latest.get('rmse', 0),
        'mae_absolute': latest.get('mae_absolute', 0),
        'rmse_absolute': latest.get('rmse_absolute', 0),
        'features_used': latest.get('features_used', 7),
        'training_size': latest.get('training_size', 'N/A')
    }

def _create_sales_plot(df, stats):
    """Создает график исторических данных"""
    try:
        plt.figure(figsize=(14, 8))
        
        if not df.empty and 'date' in df.columns and 'total_sales' in df.columns:
            # Сортируем по дате
            df = df.sort_values('date')
            
            plt.plot(df['date'], df['total_sales'], marker='o', linewidth=2, 
                    color='#2563EB', markersize=3, label='Исторические данные', alpha=0.8)
            
            # Скользящее среднее
            if len(df) >= 7:
                rolling_mean = df['total_sales'].rolling(window=7).mean()
                plt.plot(df['date'], rolling_mean, linewidth=2, color='#F97316', 
                        linestyle='--', label='Скользящее среднее (7 дней)', alpha=0.8)
            
            # Добавляем трендовую линию
            if len(df) > 10:
                x_numeric = np.arange(len(df))
                z = np.polyfit(x_numeric, df['total_sales'], 1)
                trend_line = np.poly1d(z)(x_numeric)
                plt.plot(df['date'], trend_line, linewidth=2, color='#10B981',
                        linestyle=':', label='Линейный тренд', alpha=0.7)
        
        plt.title(f'Исторические данные продаж\nВсего дней: {stats["total_days"]} | Общий объем: {stats["total_sales"]:,.0f} руб.', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Продажи (руб.)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        # Добавляем статистику на график
        stats_text = f'Максимум: {stats["max_sales"]:,.0f} руб.\nМинимум: {stats["min_sales"]:,.0f} руб.\nСреднее: {stats["avg_daily"]:,.0f} руб.\nОбщий рост: {stats["growth_rate"]:+.1f}%'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        return _plot_to_base64()
        
    except Exception as e:
        logger.error(f"Ошибка создания графика продаж: {e}")
        return None

def _create_forecast_plot(historical_data, forecast_results, model_info):
    """Создает график прогноза"""
    try:
        plt.figure(figsize=(14, 8))
        
        # Исторические данные
        if historical_data:
            df_hist = pd.DataFrame(historical_data)
            if 'date' in df_hist.columns and 'total_sales' in df_hist.columns:
                df_hist['date'] = pd.to_datetime(df_hist['date'])
                df_hist = df_hist.sort_values('date')
                plt.plot(df_hist['date'], df_hist['total_sales'], marker='o', linewidth=2,
                        color='#2563EB', markersize=3, label='Исторические данные', alpha=0.8)
        
        # Прогноз
        if forecast_results:
            dates = [pd.to_datetime(f['date']) for f in forecast_results]
            predictions = [f['predicted_sales'] for f in forecast_results]
            
            plt.plot(dates, predictions, marker='s', linewidth=3, color='#F97316',
                    markersize=6, label='Прогноз')
            
            # Доверительный интервал
            if forecast_results and 'confidence_interval' in forecast_results[0]:
                upper = [f['confidence_interval']['upper'] for f in forecast_results]
                lower = [f['confidence_interval']['lower'] for f in forecast_results]
                uncertainty = forecast_results[0]['confidence_interval']['uncertainty_pct']
                confidence_level = forecast_results[0]['confidence_interval'].get('confidence_level', 0.87)
                
                plt.fill_between(dates, lower, upper, alpha=0.3, color='#F97316',
                            label=f'Доверительный интервал ({confidence_level:.0%}) ±{uncertainty:.1f}%')
        
        plt.title(f'Прогноз продаж на 7 дней', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Продажи (руб.)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        return _plot_to_base64()
        
    except Exception as e:
        logger.error(f"Ошибка создания графика прогноза: {e}")
        return None

def _create_full_plot(historical_data, forecast_results, hist_stats, fc_stats, model_info):
    """Создает полный график"""
    try:
        plt.figure(figsize=(16, 9))
        
        # Объединяем исторические данные и прогноз
        if historical_data:
            df_hist = pd.DataFrame(historical_data)
            if 'date' in df_hist.columns and 'total_sales' in df_hist.columns:
                df_hist['date'] = pd.to_datetime(df_hist['date'])
                df_hist = df_hist.sort_values('date')
                plt.plot(df_hist['date'], df_hist['total_sales'], marker='o', linewidth=2,
                        color='#2563EB', markersize=3, label='Исторические данные', alpha=0.8)
        
        if forecast_results:
            dates = [pd.to_datetime(f['date']) for f in forecast_results]
            predictions = [f['predicted_sales'] for f in forecast_results]
            
            plt.plot(dates, predictions, marker='s', linewidth=3, color='#F97316',
                    markersize=6, label='Прогноз')
            
            if forecast_results and 'confidence_interval' in forecast_results[0]:
                upper = [f['confidence_interval']['upper'] for f in forecast_results]
                lower = [f['confidence_interval']['lower'] for f in forecast_results]
                uncertainty = forecast_results[0]['confidence_interval']['uncertainty_pct']
                
                plt.fill_between(dates, lower, upper, alpha=0.3, color='#F97316',
                               label=f'Доверительный интервал (±{uncertainty:.1f}%)')
        
        plt.title(f'Полный отчет: Исторические данные и прогноз продаж | Всего дней данных: {hist_stats["total_days"]}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Продажи (руб.)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        return _plot_to_base64()
        
    except Exception as e:
        logger.error(f"Ошибка создания полного графика: {e}")
        return None

def _plot_to_base64():
    """Конвертирует график в base64"""
    try:
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Ошибка конвертации графика: {e}")
        return None

def _html_to_pdf(html_content):
    """Конвертирует HTML в PDF"""
    if not html_content:
        return None
    
    try:
        pdf_buffer = BytesIO()
        HTML(string=html_content).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        logger.error(f"Ошибка создания PDF: {e}")
        return None

def _format_date_for_display(date_obj):
    """Форматирует дату для отображения"""
    try:
        if hasattr(date_obj, 'strftime'):
            return date_obj.strftime('%d.%m.%Y')
        elif isinstance(date_obj, str):
            # Если это строка, пытаемся преобразовать
            if len(date_obj) >= 10:
                # Пытаемся распарсить дату
                try:
                    date_parsed = datetime.datetime.strptime(date_obj[:10], '%Y-%m-%d')
                    return date_parsed.strftime('%d.%m.%Y')
                except ValueError:
                    return date_obj[:10]  # Возвращаем как есть
            return str(date_obj)
        else:
            return str(date_obj)
    except Exception as e:
        logger.warning(f"Ошибка форматирования даты {date_obj}: {e}")
        return str(date_obj)

def _get_day_name_from_date(date_obj):
    """Получает название дня недели из объекта даты"""
    try:
        if hasattr(date_obj, 'strftime'):
            date = date_obj
        else:
            # Если это строка, преобразуем в datetime
            if isinstance(date_obj, str):
                if len(date_obj) >= 10:
                    date = datetime.datetime.strptime(date_obj[:10], '%Y-%m-%d')
                else:
                    return 'Неизвестно'
            else:
                return 'Неизвестно'
        
        days = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
        return days[date.weekday()]
    except Exception as e:
        logger.warning(f"Ошибка преобразования даты {date_obj}: {e}")
        return 'Неизвестно'

def _get_day_name_from_number(day_num):
    """Получает название дня недели из номера"""
    days = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    if isinstance(day_num, (int, float)) and 0 <= int(day_num) < 7:
        return days[int(day_num)]
    return 'Н/Д'

def _get_month_name(month_num):
    """Получает название месяца из номера"""
    months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
             'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
    if isinstance(month_num, (int, float)) and 1 <= int(month_num) <= 12:
        return months[int(month_num)-1]
    return 'Н/Д'

def _render_forecast_html(forecast_results, model_info, plot_base64):
    """Рендерит улучшенный HTML для отчета по прогнозам"""
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Отчет по прогнозам</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                background: #f8fafc;
                min-height: 100vh;
            }
            .container { 
                max-width: 1200px; 
                margin: 20px auto; 
                background: white; 
                padding: 30px; 
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            }
            .header { 
                text-align: center; 
                margin-bottom: 30px; 
                padding-bottom: 25px; 
                border-bottom: 2px solid #e5e7eb;
            }
            .accuracy-info { 
                background: #f0f9ff;
                padding: 25px; 
                border-radius: 10px; 
                margin: 25px 0; 
                border-left: 4px solid #3b82f6;
            }
            .stats-grid { 
                display: grid; 
                grid-template-columns: repeat(4, 1fr); 
                gap: 12px; 
                margin: 25px 0; 
            }
            .stat-card { 
                background: white;
                padding: 15px; 
                border-radius: 8px; 
                text-align: center; 
                border: 1px solid #e5e7eb;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                min-height: 90px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .stat-value { 
                font-size: 18px; 
                font-weight: bold; 
                margin: 5px 0; 
                color: #1f2937;
                line-height: 1.2;
            }
            .stat-label { 
                font-size: 11px; 
                color: #6b7280;
                font-weight: 500;
                line-height: 1.2;
            }
            .growth-positive { 
                border-color: #10b981 !important;
                background: #f0fdf4;
            }
            .growth-negative { 
                border-color: #ef4444 !important;
                background: #fef2f2;
            }
            .growth-positive .stat-value {
                color: #059669;
            }
            .growth-negative .stat-value {
                color: #dc2626;
            }
            .table-container {
                display: flex;
                justify-content: center;
                margin: 25px 0;
            }
            table { 
                width: auto;
                border-collapse: collapse;
                font-size: 13px;
                background: white;
                margin: 0 auto;
            }
            th, td { 
                padding: 12px 15px; 
                text-align: left; 
                border-bottom: 1px solid #e5e7eb;
                white-space: nowrap;
            }
            th { 
                background: #f8fafc;
                color: #374151;
                font-weight: 600;
                border-bottom: 2px solid #e5e7eb;
            }
            tr:hover {
                background-color: #f9fafb;
            }
            img { 
                max-width: 100%; 
                height: auto; 
                margin: 25px 0; 
                border-radius: 8px;
                border: 1px solid #e5e7eb;
            }
            .footer { 
                text-align: center; 
                margin-top: 40px; 
                color: #6b7280; 
                font-size: 13px; 
                padding-top: 25px; 
                border-top: 1px solid #e5e7eb;
            }
            .data-info { 
                background: #fffbeb;
                padding: 18px; 
                border-radius: 8px; 
                margin: 20px 0; 
                border-left: 4px solid #f59e0b;
            }
            h1 {
                color: #1f2937;
                margin-bottom: 8px;
                font-weight: 700;
                font-size: 28px;
            }
            h2 {
                color: #374151;
                margin-top: 35px;
                margin-bottom: 20px;
                font-size: 22px;
                border-bottom: 1px solid #e5e7eb;
                padding-bottom: 10px;
                text-align: center;
            }
            h3 {
                color: #1f2937;
                margin-top: 0;
                margin-bottom: 15px;
            }
            .model-info-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 10px;
                margin-top: 10px;
            }
            .metric-item {
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #e5e7eb;
            }
            .metric-value {
                font-weight: 600;
                color: #1f2937;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Отчет по прогнозированию продаж</h1>
                <p style="color: #6b7280; font-size: 15px; margin-top: 5px;">Сгенерирован: {{ generation_date }}</p>
            </div>
            
            <div class="data-info">
                <strong>Период анализа:</strong> прогноз на {{ forecast_data|length }} дней с {{ format_date(forecast_data[0].date) if forecast_data else 'N/A' }} по {{ format_date(forecast_data[-1].date) if forecast_data else 'N/A' }}
            </div>
            
            <div class="accuracy-info">
                <h3>Информация о модели</h3>
                <div class="model-info-grid">
                    <div class="metric-item">
                        <span>Модель:</span>
                        <span class="metric-value">{{ model_name }}</span>
                    </div>
                    <div class="metric-item">
                        <span>Средняя абсолютная ошибка (MAE):</span>
                        <span class="metric-value">{{ "%.0f"|format(model_mae_absolute) }} руб. ({{ "%.1f"|format(model_mae) }}%)</span>
                    </div>
                    <div class="metric-item">
                        <span>Среднеквадратичная ошибка (RMSE):</span>
                        <span class="metric-value">{{ "%.0f"|format(model_rmse_absolute) }} руб. ({{ "%.1f"|format(model_rmse) }}%)</span>
                    </div>
                    <div class="metric-item">
                        <span>Количество признаков:</span>
                        <span class="metric-value">{{ features_used }}</span>
                    </div>
                    <div class="metric-item">
                        <span>Дата обучения:</span>
                        <span class="metric-value">{{ accuracy_date }}</span>
                    </div>
                </div>
            </div>
            
            {% if plot_base64 %}
            <img src="data:image/png;base64,{{ plot_base64 }}" alt="График прогноза продаж">
            {% endif %}
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ "%.0f"|format(total_forecast) }} ₽</div>
                    <div class="stat-label">Общий прогнозируемый объем</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ "%.0f"|format(avg_forecast) }} ₽</div>
                    <div class="stat-label">Среднедневной прогноз</div>
                </div>
                <div class="stat-card growth-positive">
                    <div class="stat-value">{{ "%.0f"|format(max_forecast) }} ₽</div>
                    <div class="stat-label">Максимальный прогноз</div>
                </div>
                <div class="stat-card growth-negative">
                    <div class="stat-value">{{ "%.0f"|format(min_forecast) }} ₽</div>
                    <div class="stat-label">Минимальный прогноз</div>
                </div>
                <div class="stat-card {{ 'growth-positive' if total_growth >= 0 else 'growth-negative' }}">
                    <div class="stat-value">{{ "%.1f"|format(total_growth) }}%</div>
                    <div class="stat-label">Общий рост за период</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ days_count }}</div>
                    <div class="stat-label">Дней прогноза</div>
                </div>
            </div>
            
            <h2>Детали прогноза</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Дата</th>
                            <th>Прогноз продаж</th>
                            <th>День недели</th>
                            <th>Доверительный интервал</th>
                            <th>Неопределенность</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for item in forecast_data %}
                        <tr>
                            <td><strong>{{ format_date(item.date) }}</strong></td>
                            <td style="font-weight: bold; color: #dc2626; font-size: 14px;">{{ "%.0f"|format(item.predicted_sales) }} ₽</td>
                            <td>{{ get_day_name(item.date) }}</td>
                            <td style="color: #6b7280; font-size: 12px;">
                                {{ "%.0f"|format(item.confidence_interval.lower) }} - {{ "%.0f"|format(item.confidence_interval.upper) }} ₽
                            </td>
                            <td><span style="color: #ef4444; font-weight: bold; font-size: 12px;">±{{ "%.1f"|format(item.confidence_interval.uncertainty_pct) }}%</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>Отчет сгенерирован системой прогнозирования продаж <strong>DataPulse</strong></p>
                <p>Использована модель: <strong>{{ model_name }}</strong> | MAE: <strong>{{ "%.2f"|format(model_mae) }}</strong> | RMSE: <strong>{{ "%.2f"|format(model_rmse) }}</strong></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Вычисляем статистику прогноза
    forecast_stats = _calculate_forecast_statistics(forecast_results)
    
    template = Template(template_str)
    return template.render(
        generation_date=datetime.datetime.now().strftime('%d.%m.%Y в %H:%M'),
        forecast_data=forecast_results,
        plot_base64=plot_base64,
        format_date=_format_date_for_display,
        get_day_name=_get_day_name_from_date,
        model_name=model_info['name'],
        model_mae=model_info.get('mae', 0),
        model_rmse=model_info.get('rmse', 0),
        model_mae_absolute=model_info.get('mae_absolute', 0),
        model_rmse_absolute=model_info.get('rmse_absolute', 0),
        features_used=model_info.get('features_used', 7),
        training_size=model_info.get('training_size', 'N/A'),
        accuracy_date=model_info['date'],
        total_forecast=forecast_stats.get('total_forecast', 0),
        avg_forecast=forecast_stats.get('avg_forecast', 0),
        max_forecast=forecast_stats.get('max_forecast', 0),
        min_forecast=forecast_stats.get('min_forecast', 0),
        total_growth=forecast_stats.get('total_growth', 0),
        days_count=forecast_stats.get('days_count', 0)
    )

def _render_sales_html(stats, historical_data, plot_base64):
    """Рендерит улучшенный HTML для отчета по продажам"""
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Отчет по продажам</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                background: #f8fafc;
                min-height: 100vh;
            }
            .container { 
                max-width: 1200px; 
                margin: 20px auto; 
                background: white; 
                padding: 30px; 
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            }
            .header { 
                text-align: center; 
                margin-bottom: 30px; 
                padding-bottom: 25px; 
                border-bottom: 2px solid #e5e7eb;
            }
            .stats-grid { 
                display: grid; 
                grid-template-columns: repeat(4, 1fr); 
                gap: 12px; 
                margin: 25px 0; 
            }
            .stat-card { 
                background: white;
                padding: 15px; 
                border-radius: 8px; 
                text-align: center; 
                border: 1px solid #e5e7eb;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                min-height: 90px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .stat-value { 
                font-size: 18px; 
                font-weight: bold; 
                margin: 5px 0; 
                color: #1f2937;
                line-height: 1.2;
            }
            .stat-label { 
                font-size: 11px; 
                color: #6b7280;
                font-weight: 500;
                line-height: 1.2;
            }
            .growth-positive { 
                border-color: #10b981 !important;
                background: #f0fdf4;
            }
            .growth-negative { 
                border-color: #ef4444 !important;
                background: #fef2f2;
            }
            .growth-positive .stat-value {
                color: #059669;
            }
            .growth-negative .stat-value {
                color: #dc2626;
            }
            table { 
                width: 100%; 
                border-collapse: collapse;
                margin: 25px 0; 
                font-size: 13px;
                background: white;
            }
            th, td { 
                padding: 12px 15px; 
                text-align: left; 
                border-bottom: 1px solid #e5e7eb;
            }
            th { 
                background: #f8fafc;
                color: #374151;
                font-weight: 600;
                border-bottom: 2px solid #e5e7eb;
            }
            tr:hover {
                background-color: #f9fafb;
            }
            img { 
                max-width: 100%; 
                height: auto; 
                margin: 25px 0; 
                border-radius: 8px;
                border: 1px solid #e5e7eb;
            }
            .footer { 
                text-align: center; 
                margin-top: 40px; 
                color: #6b7280; 
                font-size: 13px; 
                padding-top: 25px; 
                border-top: 1px solid #e5e7eb;
            }
            .highlight { 
                background: #fffbeb;
                padding: 18px; 
                border-radius: 8px; 
                margin: 20px 0; 
                border-left: 4px solid #f59e0b;
            }
            h1 {
                color: #1f2937;
                margin-bottom: 8px;
                font-weight: 700;
                font-size: 28px;
            }
            h2 {
                color: #374151;
                margin-top: 35px;
                margin-bottom: 20px;
                font-size: 22px;
                border-bottom: 1px solid #e5e7eb;
                padding-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Отчет по историческим данным продаж</h1>
                <p style="color: #6b7280; font-size: 15px; margin-top: 5px;">Сгенерирован: {{ generation_date }}</p>
            </div>
            
            <div class="highlight">
                <strong>Обзор данных:</strong> Анализ продаж за весь период с {{ format_date(historical_data[0].date) if historical_data else 'N/A' }} по {{ format_date(historical_data[-1].date) if historical_data else 'N/A' }}
            </div>
            
            {% if plot_base64 %}
            <img src="data:image/png;base64,{{ plot_base64 }}" alt="График исторических данных продаж">
            {% endif %}
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ "%.0f"|format(stats.total_sales) }} ₽</div>
                    <div class="stat-label">Общий объем продаж</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ stats.total_days }}</div>
                    <div class="stat-label">Дней анализа</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ "%.0f"|format(stats.avg_daily) }} ₽</div>
                    <div class="stat-label">Среднедневной объем</div>
                </div>
                <div class="stat-card {{ 'growth-positive' if stats.growth_rate >= 0 else 'growth-negative' }}">
                    <div class="stat-value">{{ "%.1f"|format(stats.growth_rate) }}%</div>
                    <div class="stat-label">Общий рост</div>
                </div>
                <div class="stat-card growth-positive">
                    <div class="stat-value">{{ "%.0f"|format(stats.max_sales) }} ₽</div>
                    <div class="stat-label">Максимальные продажи</div>
                    <div style="font-size: 9px; color: #6b7280; margin-top: 4px;">{{ stats.best_day_date }}</div>
                </div>
                <div class="stat-card growth-negative">
                    <div class="stat-value">{{ "%.0f"|format(stats.min_sales) }} ₽</div>
                    <div class="stat-label">Минимальные продажи</div>
                    <div style="font-size: 9px; color: #6b7280; margin-top: 4px;">{{ stats.worst_day_date }}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ "%.0f"|format(stats.std_sales) }} ₽</div>
                    <div class="stat-label">Стандартное отклонение</div>
                </div>
                <div class="stat-card {{ 'growth-positive' if stats.avg_growth >= 0 else 'growth-negative' }}">
                    <div class="stat-value">{{ "%.2f"|format(stats.avg_growth) }}%</div>
                    <div class="stat-label">Средний дневной рост</div>
                </div>
            </div>
            
            <h2>Исторические данные (первые 20 записей)</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 15%;">Дата</th>
                        <th style="width: 20%;">Продажи</th>
                        <th style="width: 20%;">День недели</th>
                        <th style="width: 20%;">Месяц</th>
                        <th style="width: 15%;">Выходной</th>
                    </tr>
                </thead>
                <tbody>
                    {% for item in historical_data[:20] %}
                    <tr>
                        <td><strong>{{ format_date(item.date) }}</strong></td>
                        <td style="font-weight: bold; color: #059669;">{{ "%.0f"|format(item.total_sales) }} ₽</td>
                        <td>{{ get_day_name_historical(item.day_of_week) }}</td>
                        <td>{{ get_month_name(item.month) }}</td>
                        <td><span style="color: {{ '#ef4444' if item.is_weekend else '#059669' }}; font-weight: bold;">
                            {{ 'Да' if item.is_weekend else 'Нет' }}
                        </span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            {% if historical_data|length > 20 %}
            <p style="text-align: center; color: #6b7280; font-style: italic; background: #f8fafc; padding: 12px; border-radius: 6px;">
                ... и еще {{ historical_data|length - 20 }} записей
            </p>
            {% endif %}
            
            <div class="footer">
                <p>Отчет сгенерирован системой прогнозирования продаж <strong>DataPulse</strong></p>
                <p>Всего записей в базе: <strong>{{ historical_data|length }}</strong></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    template = Template(template_str)
    return template.render(
        generation_date=datetime.datetime.now().strftime('%d.%m.%Y в %H:%M'),
        stats=stats,
        historical_data=historical_data,
        plot_base64=plot_base64,
        format_date=_format_date_for_display,
        get_day_name_historical=_get_day_name_from_number,
        get_month_name=_get_month_name
    )

def _render_full_html(historical_data, forecast_results, hist_stats, fc_stats, model_info, plot_base64):
    """Рендерит HTML для полного отчета"""
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Полный отчет по прогнозированию</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                background: #f8fafc;
                min-height: 100vh;
            }
            .container { 
                max-width: 1200px; 
                margin: 20px auto; 
                background: white; 
                padding: 30px; 
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            }
            .header { 
                text-align: center; 
                margin-bottom: 30px; 
                padding-bottom: 25px; 
                border-bottom: 2px solid #e5e7eb;
            }
            .model-info { 
                background: #f0f9ff;
                padding: 25px; 
                border-radius: 10px; 
                margin: 25px 0; 
                border-left: 4px solid #3b82f6;
            }
            .stats-grid { 
                display: grid; 
                grid-template-columns: repeat(4, 1fr); 
                gap: 12px; 
                margin: 25px 0; 
            }
            .stat-card { 
                background: white;
                padding: 15px; 
                border-radius: 8px; 
                text-align: center; 
                border: 1px solid #e5e7eb;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                min-height: 90px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .stat-value { 
                font-size: 18px; 
                font-weight: bold; 
                margin: 5px 0; 
                color: #1f2937;
                line-height: 1.2;
            }
            .stat-label { 
                font-size: 11px; 
                color: #6b7280;
                font-weight: 500;
                line-height: 1.2;
            }
            .growth-positive { 
                border-color: #10b981 !important;
                background: #f0fdf4;
            }
            .growth-negative { 
                border-color: #ef4444 !important;
                background: #fef2f2;
            }
            .growth-positive .stat-value {
                color: #059669;
            }
            .growth-negative .stat-value {
                color: #dc2626;
            }
            .table-container {
                display: flex;
                justify-content: center;
                margin: 25px 0;
            }
            table { 
                width: auto;
                border-collapse: collapse;
                font-size: 13px;
                background: white;
                margin: 0 auto;
            }
            th, td { 
                padding: 12px 15px; 
                text-align: left; 
                border-bottom: 1px solid #e5e7eb;
                white-space: nowrap;
            }
            th { 
                background: #f8fafc;
                color: #374151;
                font-weight: 600;
                border-bottom: 2px solid #e5e7eb;
            }
            tr:hover {
                background-color: #f9fafb;
            }
            img { 
                max-width: 100%; 
                height: auto; 
                margin: 25px 0; 
                border-radius: 8px;
                border: 1px solid #e5e7eb;
            }
            .footer { 
                text-align: center; 
                margin-top: 40px; 
                color: #6b7280; 
                font-size: 13px; 
                padding-top: 25px; 
                border-top: 1px solid #e5e7eb;
            }
            .data-info { 
                background: #fffbeb;
                padding: 18px; 
                border-radius: 8px; 
                margin: 20px 0; 
                border-left: 4px solid #f59e0b;
            }
            .summary { 
                background: #f0fdf4;
                padding: 20px; 
                border-radius: 8px; 
                margin: 20px 0; 
                border-left: 4px solid #10b981;
            }
            h1 {
                color: #1f2937;
                margin-bottom: 8px;
                font-weight: 700;
                font-size: 28px;
            }
            h2 {
                color: #374151;
                margin-top: 35px;
                margin-bottom: 20px;
                font-size: 22px;
                border-bottom: 1px solid #e5e7eb;
                padding-bottom: 10px;
                text-align: center;
            }
            h3 {
                color: #1f2937;
                margin-top: 0;
                margin-bottom: 15px;
            }
            .model-info-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 10px;
                margin-top: 10px;
            }
            .metric-item {
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #e5e7eb;
            }
            .metric-value {
                font-weight: 600;
                color: #1f2937;
            }
            .comparison-section {
                background: #f8fafc;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border: 1px solid #e5e7eb;
            }
            .vertical-stats {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }
            .stats-column {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .column-title {
                text-align: center;
                font-weight: 600;
                color: #374151;
                margin-bottom: 15px;
                font-size: 16px;
                padding: 10px;
                border-radius: 6px;
            }
            .historical-title {
                background: #dbeafe;
                color: #1e40af;
            }
            .forecast-title {
                background: #fed7aa;
                color: #c2410c;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Полный отчет по прогнозированию продаж</h1>
                <p style="color: #6b7280; font-size: 15px; margin-top: 5px;">Сгенерирован: {{ generation_date }}</p>
            </div>
            
            <div class="data-info">
                <strong>Период анализа:</strong> 
                Исторические данные: {{ hist_stats.total_days }} дней | 
                Прогноз: {{ fc_stats.days_count }} дней с {{ format_date(forecast_data[0].date) if forecast_data else 'N/A' }}
            </div>
            
            <div class="model-info">
                <h3>Информация о модели</h3>
                <div class="model-info-grid">
                    <div class="metric-item">
                        <span>Модель:</span>
                        <span class="metric-value">{{ model_name }}</span>
                    </div>
                    <div class="metric-item">
                        <span>Средняя абсолютная ошибка (MAE):</span>
                        <span class="metric-value">{{ "%.0f"|format(model_mae_absolute) }} руб. ({{ "%.1f"|format(model_mae) }}%)</span>
                    </div>
                    <div class="metric-item">
                        <span>Среднеквадратичная ошибка (RMSE):</span>
                        <span class="metric-value">{{ "%.0f"|format(model_rmse_absolute) }} руб. ({{ "%.1f"|format(model_rmse) }}%)</span>
                    </div>
                    <div class="metric-item">
                        <span>Количество признаков:</span>
                        <span class="metric-value">{{ features_used }}</span>
                    </div>
                    <div class="metric-item">
                        <span>Дата обучения:</span>
                        <span class="metric-value">{{ accuracy_date }}</span>
                    </div>
                </div>
            </div>
            
            <div class="summary">
                <h3>Краткая сводка</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <p><strong>Общий объем продаж за исторический период:</strong><br>{{ "%.0f"|format(hist_stats.total_sales) }} руб.</p>
                        <p><strong>Среднедневные продажи:</strong><br>{{ "%.0f"|format(hist_stats.avg_daily) }} руб.</p>
                        <p><strong>Общий рост за исторический период:</strong><br>
                            <span style="color: {{ '#059669' if hist_stats.growth_rate >= 0 else '#dc2626' }}; font-weight: bold;">
                                {{ "%.1f"|format(hist_stats.growth_rate) }}%
                            </span>
                        </p>
                    </div>
                    <div>
                        <p><strong>Прогнозируемый объем на {{ fc_stats.days_count }} дней:</strong><br>{{ "%.0f"|format(fc_stats.total_forecast) }} руб.</p>
                        <p><strong>Среднедневной прогноз:</strong><br>{{ "%.0f"|format(fc_stats.avg_forecast) }} руб.</p>
                        <p><strong>Рост за период прогноза:</strong><br>
                            <span style="color: {{ '#059669' if fc_stats.total_growth >= 0 else '#dc2626' }}; font-weight: bold;">
                                {{ "%.1f"|format(fc_stats.total_growth) }}%
                            </span>
                        </p>
                    </div>
                </div>
            </div>
            
            {% if plot_base64 %}
            <img src="data:image/png;base64,{{ plot_base64 }}" alt="Полный график данных и прогноза">
            {% endif %}
            
            <div class="comparison-section">
                <h3 style="text-align: center;">Сравнительная статистика</h3>
                <div class="vertical-stats">
                    <!-- Исторические данные - вертикальный столбец -->
                    <div class="stats-column">
                        <div class="column-title historical-title">Исторические данные</div>
                        <div class="stat-card">
                            <div class="stat-value">{{ "%.0f"|format(hist_stats.total_sales) }} ₽</div>
                            <div class="stat-label">Общий объем продаж</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{{ hist_stats.total_days }}</div>
                            <div class="stat-label">Дней анализа</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{{ "%.0f"|format(hist_stats.avg_daily) }} ₽</div>
                            <div class="stat-label">Среднедневной объем</div>
                        </div>
                        <div class="stat-card {{ 'growth-positive' if hist_stats.growth_rate >= 0 else 'growth-negative' }}">
                            <div class="stat-value">{{ "%.1f"|format(hist_stats.growth_rate) }}%</div>
                            <div class="stat-label">Общий рост</div>
                        </div>
                        <div class="stat-card growth-positive">
                            <div class="stat-value">{{ "%.0f"|format(hist_stats.max_sales) }} ₽</div>
                            <div class="stat-label">Максимальные продажи</div>
                            <div style="font-size: 9px; color: #6b7280; margin-top: 4px;">{{ hist_stats.best_day_date }}</div>
                        </div>
                        <div class="stat-card growth-negative">
                            <div class="stat-value">{{ "%.0f"|format(hist_stats.min_sales) }} ₽</div>
                            <div class="stat-label">Минимальные продажи</div>
                            <div style="font-size: 9px; color: #6b7280; margin-top: 4px;">{{ hist_stats.worst_day_date }}</div>
                        </div>
                    </div>
                    
                    <!-- Прогноз - вертикальный столбец -->
                    <div class="stats-column">
                        <div class="column-title forecast-title">Прогноз на {{ fc_stats.days_count }} дней</div>
                        <div class="stat-card">
                            <div class="stat-value">{{ "%.0f"|format(fc_stats.total_forecast) }} ₽</div>
                            <div class="stat-label">Общий прогнозируемый объем</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{{ fc_stats.days_count }}</div>
                            <div class="stat-label">Дней прогноза</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{{ "%.0f"|format(fc_stats.avg_forecast) }} ₽</div>
                            <div class="stat-label">Среднедневной прогноз</div>
                        </div>
                        <div class="stat-card {{ 'growth-positive' if fc_stats.total_growth >= 0 else 'growth-negative' }}">
                            <div class="stat-value">{{ "%.1f"|format(fc_stats.total_growth) }}%</div>
                            <div class="stat-label">Рост за период</div>
                        </div>
                        <div class="stat-card growth-positive">
                            <div class="stat-value">{{ "%.0f"|format(fc_stats.max_forecast) }} ₽</div>
                            <div class="stat-label">Максимальный прогноз</div>
                        </div>
                        <div class="stat-card growth-negative">
                            <div class="stat-value">{{ "%.0f"|format(fc_stats.min_forecast) }} ₽</div>
                            <div class="stat-label">Минимальный прогноз</div>
                        </div>
                    </div>
                </div>
            </div>

            {% if fc_stats.total_growth > 0 %}
            <div style="background: #f0fdf4; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #10b981;">
                <strong style="color: #059669;">Положительная динамика:</strong> Прогнозируется рост продаж на {{ "%.1f"|format(fc_stats.total_growth) }}% за период прогноза.
            </div>
            {% elif fc_stats.total_growth < 0 %}
            <div style="background: #fef2f2; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #dc2626;">
                <strong style="color: #dc2626;">Отрицательная динамика:</strong> Прогнозируется снижение продаж на {{ "%.1f"|format(fc_stats.total_growth|abs) }}% за период прогноза.
            </div>
            {% endif %}
            
            <h2>Детали прогноза</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Дата</th>
                            <th>Прогноз продаж</th>
                            <th>День недели</th>
                            <th>Доверительный интервал</th>
                            <th>Неопределенность</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for item in forecast_data %}
                        <tr>
                            <td><strong>{{ format_date(item.date) }}</strong></td>
                            <td style="font-weight: bold; color: #dc2626; font-size: 14px;">{{ "%.0f"|format(item.predicted_sales) }} ₽</td>
                            <td>{{ get_day_name(item.date) }}</td>
                            <td style="color: #6b7280; font-size: 12px;">
                                {{ "%.0f"|format(item.confidence_interval.lower) }} - {{ "%.0f"|format(item.confidence_interval.upper) }} ₽
                            </td>
                            <td><span style="color: #ef4444; font-weight: bold; font-size: 12px;">±{{ "%.1f"|format(item.confidence_interval.uncertainty_pct) }}%</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Дата</th>
                            <th>Продажи</th>
                            <th>День недели</th>
                            <th>Месяц</th>
                            <th>Выходной</th>
                            <th>Тип дня</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for item in historical_data %}
                        <tr>
                            <td><strong>{{ format_date(item.date) }}</strong></td>
                            <td style="font-weight: bold; color: #059669;">{{ "%.0f"|format(item.total_sales) }} ₽</td>
                            <td>{{ get_day_name_historical(item.day_of_week) }}</td>
                            <td>{{ get_month_name(item.month) }}</td>
                            <td><span style="color: {{ '#ef4444' if item.is_weekend else '#059669' }}; font-weight: bold;">
                                {{ 'Да' if item.is_weekend else 'Нет' }}
                            </span></td>
                            <td>
                                {% if item.is_holiday %}
                                    <span style="color: #f59e0b; font-weight: bold;">Праздник</span>
                                {% elif item.is_weekend %}
                                    <span style="color: #ef4444; font-weight: bold;">Выходной</span>
                                {% else %}
                                    <span style="color: #6b7280;">Будний</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>Отчет сгенерирован системой прогнозирования продаж <strong>DataPulse</strong></p>
                <p>Использована модель: <strong>{{ model_name }}</strong> | MAE: <strong>{{ "%.2f"|format(model_mae) }}</strong> | RMSE: <strong>{{ "%.2f"|format(model_rmse) }}</strong></p>
                <p>Исторических записей: <strong>{{ historical_data|length }}</strong> | Прогнозов: <strong>{{ forecast_data|length }}</strong></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    template = Template(template_str)
    return template.render(
        generation_date=datetime.datetime.now().strftime('%d.%m.%Y в %H:%M'),
        forecast_data=forecast_results,
        historical_data=historical_data,
        plot_base64=plot_base64,
        format_date=_format_date_for_display,
        get_day_name=_get_day_name_from_date,
        get_day_name_historical=_get_day_name_from_number,
        get_month_name=_get_month_name,
        model_name=model_info['name'],
        model_mae=model_info.get('mae', 0),
        model_rmse=model_info.get('rmse', 0),
        model_mae_absolute=model_info.get('mae_absolute', 0),
        model_rmse_absolute=model_info.get('rmse_absolute', 0),
        features_used=model_info.get('features_used', 7),
        accuracy_date=model_info['date'],
        hist_stats=hist_stats,
        fc_stats=fc_stats
    )

if __name__ == "__main__":
    print("Модуль генерации отчетов загружен")
    
    