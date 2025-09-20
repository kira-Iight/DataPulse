import pandas as pd
import numpy as np
from jinja2 import Template
from weasyprint import HTML
import datetime
import matplotlib
matplotlib.use('Agg')  # Устанавливаем backend для работы без GUI
import matplotlib.pyplot as plt
import os
import base64
import sys
from io import BytesIO
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

def generate_sales_report(session_data):
    """Генерирует отчет только по историческим данным продаж."""
    try:
        processed_data = session_data.get('processed_data', [])
        
        if not processed_data:
            print("Нет исторических данных для отчета")
            return None
            
        df_data = pd.DataFrame(processed_data)
        
        # Вычисляем статистику
        if not df_data.empty and 'total_sales' in df_data.columns and 'date' in df_data.columns:
            try:
                df_data['date'] = pd.to_datetime(df_data['date'])
                total_sales = float(df_data['total_sales'].sum())
                avg_daily = float(df_data['total_sales'].mean())
                max_sales = float(df_data['total_sales'].max())
                min_sales = float(df_data['total_sales'].min())
                std_sales = float(df_data['total_sales'].std())
                if len(df_data) > 1 and df_data['total_sales'].iloc[0] != 0:
                    growth_rate = float((df_data['total_sales'].iloc[-1] - df_data['total_sales'].iloc[0]) / df_data['total_sales'].iloc[0] * 100)
                else:
                    growth_rate = 0.0
            except Exception as e:
                print(f"Ошибка при вычислении статистики: {e}")
                total_sales = avg_daily = max_sales = min_sales = std_sales = growth_rate = 0.0
        else:
            total_sales = avg_daily = max_sales = min_sales = std_sales = growth_rate = 0.0
        
        # Создаем график исторических данных
        try:
            plt.figure(figsize=(14, 8))
            
            if not df_data.empty and 'date' in df_data.columns and 'total_sales' in df_data.columns:
                dates_data = df_data['date']
                sales_data = df_data['total_sales']
                
                # Основная линия продаж
                plt.plot(dates_data, sales_data, label='Исторические данные', marker='o', linewidth=3, color='#2563EB', markersize=6)
                
                # Добавляем скользящее среднее
                if len(sales_data) >= 7:
                    rolling_mean = sales_data.rolling(window=7).mean()
                    plt.plot(dates_data, rolling_mean, label='Скользящее среднее (7 дней)', linewidth=2, color='#F97316', linestyle='--')
                
                # Добавляем тренд
                if len(sales_data) > 1:
                    try:
                        z = np.polyfit(range(len(sales_data)), sales_data, 1)
                        p = np.poly1d(z)
                        plt.plot(dates_data, p(range(len(sales_data))), label='Тренд', linewidth=2, color='#10B981', linestyle=':')
                    except Exception as e:
                        print(f"Ошибка при создании тренда: {e}")
            
            # Настройки графика
            plt.title(f'Исторические данные продаж\nОбщий объем: {total_sales:,.0f} руб. | Среднедневной: {avg_daily:,.0f} руб.', fontsize=16, fontweight='bold')
            plt.xlabel('Дата', fontsize=12)
            plt.ylabel('Продажи (руб.)', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.gcf().autofmt_xdate()
            
            # Добавляем статистику на график
            if not df_data.empty:
                stats_text = f'Максимум: {max_sales:,.0f} руб.\nМинимум: {min_sales:,.0f} руб.\nСтандартное отклонение: {std_sales:,.0f} руб.\nРост: {growth_rate:+.1f}%'
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Сохраняем график в буфер
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            
            # Конвертируем изображение в base64
            plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            print(f"Ошибка при создании графика: {e}")
            # Создаем простой график в случае ошибки
            plt.figure(figsize=(14, 8))
            plt.text(0.5, 0.5, 'Ошибка при создании графика', ha='center', va='center', fontsize=16)
            plt.title('Исторические данные продаж', fontsize=16)
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Создаем HTML отчет
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sales Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }
                .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
                h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                .stat-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #dee2e6; }
                .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .stat-label { color: #6c757d; font-size: 14px; }
                .growth-positive { color: #27ae60; }
                .growth-negative { color: #e74c3c; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; font-weight: 600; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                img { max-width: 100%; height: auto; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .footer { text-align: center; margin-top: 30px; color: #6c757d; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📈 Отчет по историческим данным продаж</h1>
                <p style="text-align: center; color: #6c757d;">Сгенерирован: {{ generation_date }}</p>
                
                <img src="data:image/png;base64,{{ plot_base64 }}" alt="График продаж">
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.0f"|format(total_sales) }}</div>
                        <div class="stat-label">Общий объем (руб.)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.0f"|format(avg_daily) }}</div>
                        <div class="stat-label">Среднедневной (руб.)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.0f"|format(max_sales) }}</div>
                        <div class="stat-label">Максимум (руб.)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.0f"|format(min_sales) }}</div>
                        <div class="stat-label">Минимум (руб.)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.0f"|format(std_sales) }}</div>
                        <div class="stat-label">Стандартное отклонение (руб.)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value {{ 'growth-positive' if growth_rate >= 0 else 'growth-negative' }}">{{ "%.1f"|format(growth_rate) }}%</div>
                        <div class="stat-label">Общий рост</div>
                    </div>
                </div>
                
                <h2>📅 Исторические данные</h2>
                <table>
                    <tr>
                        <th>Дата</th>
                        <th>Продажи (руб.)</th>
                        <th>День недели</th>
                        <th>Месяц</th>
                        <th>Праздничный день</th>
                    </tr>
                    {% for item in historical_data %}
                    <tr>
                        <td>{{ item.date }}</td>
                        <td style="font-weight: bold; color: #2c3e50;">{{ "%.2f"|format(item.total_sales) }}</td>
                        <td>{{ get_day_name(item.day_of_week) }}</td>
                        <td>{{ get_month_name(item.month) }}</td>
                        <td>{{ 'Да' if item.is_holiday else 'Нет' }}</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <div class="footer">
                    <p>Отчет сгенерирован системой прогнозирования продаж DataPulse Analytics</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Функции для получения названий дней и месяцев
        def get_day_name(day_number):
            days = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
            return days[day_number] if 0 <= day_number < 7 else 'Неизвестно'
        
        def get_month_name(month_number):
            months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                     'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
            return months[month_number - 1] if 1 <= month_number <= 12 else 'Неизвестно'
        
        template = Template(html_template)
        html_content = template.render(
            generation_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            historical_data=processed_data,
            plot_base64=plot_base64,
            get_day_name=get_day_name,
            get_month_name=get_month_name,
            total_sales=total_sales,
            avg_daily=avg_daily,
            max_sales=max_sales,
            min_sales=min_sales,
            std_sales=std_sales,
            growth_rate=growth_rate
        )
        
        # Конвертируем HTML в PDF
        try:
            pdf_buffer = BytesIO()
            HTML(string=html_content).write_pdf(pdf_buffer)
            pdf_buffer.seek(0)
            return pdf_buffer
        except Exception as e:
            print(f"Ошибка при создании PDF для отчета по продажам: {e}")
            return None
        
    except Exception as e:
        print(f"Ошибка при генерации отчета по продажам: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_forecast_report(session_data):
    """Генерирует отчет только по прогнозам."""
    try:
        forecast_results = session_data.get('forecast_results', [])
        model_accuracy = session_data.get('model_accuracy', [])
        
        if not forecast_results:
            print("Нет данных прогноза для отчета")
            return None
            
        df_forecast = pd.DataFrame(forecast_results)
        
        # Получаем точность модели
        accuracy_info = model_accuracy[-1] if model_accuracy else None
        accuracy_value = accuracy_info.get('accuracy', 0) if accuracy_info else 0
        model_name = accuracy_info.get('model_name', 'Unknown') if accuracy_info else 'Unknown'
        
        # Создаем график прогнозов
        try:
            plt.figure(figsize=(14, 8))
            
            if not df_forecast.empty and 'date' in df_forecast.columns and 'predicted_sales' in df_forecast.columns:
                df_forecast['date'] = pd.to_datetime(df_forecast['date'])
                dates_forecast = df_forecast['date']
                sales_forecast = df_forecast['predicted_sales']
                
                # Основная линия прогноза
                plt.plot(dates_forecast, sales_forecast, label='Прогноз продаж', marker='s', linestyle='-', linewidth=3, color='#F97316')
                
                # Добавляем доверительный интервал (±20%)
                upper_bound = sales_forecast * 1.2
                lower_bound = sales_forecast * 0.8
                plt.fill_between(dates_forecast, lower_bound, upper_bound, alpha=0.3, color='#F97316', label='Доверительный интервал (±20%)')
            
            # Настройки графика
            plt.title(f'Прогноз продаж на 7 дней\nТочность модели: {model_name} - {(1-accuracy_value)*100:.1f}%', fontsize=16, fontweight='bold')
            plt.xlabel('Дата', fontsize=12)
            plt.ylabel('Продажи (руб.)', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.gcf().autofmt_xdate()
            
            # Добавляем статистику
            if not df_forecast.empty:
                total_forecast = float(sales_forecast.sum())
                avg_daily = float(sales_forecast.mean())
                max_forecast = float(sales_forecast.max())
                min_forecast = float(sales_forecast.min())
                
                stats_text = f'Общий прогноз: {total_forecast:,.0f} руб.\nСреднедневной: {avg_daily:,.0f} руб.\nМаксимум: {max_forecast:,.0f} руб.\nМинимум: {min_forecast:,.0f} руб.'
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Сохраняем график в буфер
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            
            # Конвертируем изображение в base64
            plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            print(f"Ошибка при создании графика прогноза: {e}")
            # Создаем простой график в случае ошибки
            plt.figure(figsize=(14, 8))
            plt.text(0.5, 0.5, 'Ошибка при создании графика прогноза', ha='center', va='center', fontsize=16)
            plt.title('Прогноз продаж на 7 дней', fontsize=16)
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Устанавливаем значения по умолчанию
            total_forecast = avg_daily = max_forecast = min_forecast = 0.0
        
        # Создаем HTML отчет
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forecast Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }
                .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
                h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                .accuracy-info { background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #27ae60; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                .stat-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #dee2e6; }
                .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .stat-label { color: #6c757d; font-size: 14px; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; font-weight: 600; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                img { max-width: 100%; height: auto; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .footer { text-align: center; margin-top: 30px; color: #6c757d; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Отчет по прогнозированию продаж</h1>
                <p style="text-align: center; color: #6c757d;">Сгенерирован: {{ generation_date }}</p>
                
                <div class="accuracy-info">
                    <h3 style="margin-top: 0; color: #27ae60;">🎯 Информация о модели</h3>
                    <p><strong>Модель:</strong> {{ model_name }}</p>
                    <p><strong>Точность:</strong> {{ "%.1f"|format((1-accuracy_value)*100) }}%</p>
                    <p><strong>Дата обучения:</strong> {{ accuracy_date }}</p>
                </div>
                
                <img src="data:image/png;base64,{{ plot_base64 }}" alt="График прогноза продаж">
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.0f"|format(total_forecast) }}</div>
                        <div class="stat-label">Общий прогноз (руб.)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.0f"|format(avg_daily) }}</div>
                        <div class="stat-label">Среднедневной (руб.)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.0f"|format(max_forecast) }}</div>
                        <div class="stat-label">Максимум (руб.)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ "%.0f"|format(min_forecast) }}</div>
                        <div class="stat-label">Минимум (руб.)</div>
                    </div>
                </div>
                
                <h2>📅 Детали прогноза</h2>
                <table>
                    <tr>
                        <th>Дата</th>
                        <th>Прогноз продаж (руб.)</th>
                        <th>День недели</th>
                        <th>Доверительный интервал</th>
                    </tr>
                    {% for item in forecast_data %}
                    <tr>
                        <td>{{ item.date }}</td>
                        <td style="font-weight: bold; color: #2c3e50;">{{ "%.2f"|format(item.predicted_sales) }}</td>
                        <td>{{ get_day_name(item.date) }}</td>
                        <td>±20% ({{ "%.2f"|format(item.predicted_sales * 0.8) }} - {{ "%.2f"|format(item.predicted_sales * 1.2) }})</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <div class="footer">
                    <p>Отчет сгенерирован системой прогнозирования продаж DataPulse Analytics</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Функция для получения названия дня недели
        def get_day_name(date_str):
            try:
                date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                days = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
                return days[date_obj.weekday()]
            except:
                return 'Неизвестно'
        
        template = Template(html_template)
        html_content = template.render(
            generation_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            forecast_data=forecast_results,
            plot_base64=plot_base64,
            get_day_name=get_day_name,
            model_name=model_name,
            accuracy_value=accuracy_value,
            accuracy_date=accuracy_info.get('created_at', 'Неизвестно')[:10] if accuracy_info else 'Неизвестно',
            total_forecast=total_forecast if not df_forecast.empty else 0,
            avg_daily=avg_daily if not df_forecast.empty else 0,
            max_forecast=max_forecast if not df_forecast.empty else 0,
            min_forecast=min_forecast if not df_forecast.empty else 0
        )
        
        # Конвертируем HTML в PDF
        try:
            pdf_buffer = BytesIO()
            HTML(string=html_content).write_pdf(pdf_buffer)
            pdf_buffer.seek(0)
            return pdf_buffer
        except Exception as e:
            print(f"Ошибка при создании PDF для отчета по прогнозам: {e}")
            return None
        
    except Exception as e:
        print(f"Ошибка при генерации отчета по прогнозам: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def generate_full_report(session_data):
    """Генерирует PDF отчет из данных сессии."""
    try:
        # Получаем данные из сессии
        processed_data = session_data.get('processed_data', [])
        forecast_results = session_data.get('forecast_results', [])
        model_accuracy = session_data.get('model_accuracy', [])
        
        df_data = pd.DataFrame(processed_data)
        df_forecast = pd.DataFrame(forecast_results)
        
        # Получаем информацию о точности модели
        accuracy_info = model_accuracy[-1] if model_accuracy else None
        accuracy_value = accuracy_info.get('accuracy', 0) if accuracy_info else 0
        model_name = accuracy_info.get('model_name', 'Unknown') if accuracy_info else 'Unknown'
        
        # Вычисляем статистику для исторических данных
        if not df_data.empty and 'total_sales' in df_data.columns and 'date' in df_data.columns:
            try:
                df_data['date'] = pd.to_datetime(df_data['date'])
                total_sales = float(df_data['total_sales'].sum())
                avg_daily = float(df_data['total_sales'].mean())
                max_sales = float(df_data['total_sales'].max())
                min_sales = float(df_data['total_sales'].min())
            except Exception as e:
                print(f"Ошибка при вычислении статистики исторических данных: {e}")
                total_sales = avg_daily = max_sales = min_sales = 0.0
        else:
            total_sales = avg_daily = max_sales = min_sales = 0.0
        
        # Вычисляем статистику для прогноза
        if not df_forecast.empty and 'predicted_sales' in df_forecast.columns:
            try:
                total_forecast = float(df_forecast['predicted_sales'].sum())
                avg_forecast = float(df_forecast['predicted_sales'].mean())
                max_forecast = float(df_forecast['predicted_sales'].max())
                min_forecast = float(df_forecast['predicted_sales'].min())
            except Exception as e:
                print(f"Ошибка при вычислении статистики прогноза: {e}")
                total_forecast = avg_forecast = max_forecast = min_forecast = 0.0
        else:
            total_forecast = avg_forecast = max_forecast = min_forecast = 0.0
        
        # Создаем график
        try:
            plt.figure(figsize=(16, 10))
            
            # Исторические данные
            if not df_data.empty and 'date' in df_data.columns and 'total_sales' in df_data.columns:
                dates_data = df_data['date']
                sales_data = df_data['total_sales']
                plt.plot(dates_data, sales_data, label='Исторические данные', marker='o', linewidth=3, color='#2563EB', markersize=6)
                
                # Добавляем скользящее среднее для исторических данных
                if len(sales_data) >= 7:
                    rolling_mean = sales_data.rolling(window=7).mean()
                    plt.plot(dates_data, rolling_mean, label='Скользящее среднее (7 дней)', linewidth=2, color='#3498db', linestyle='--')
            
            # Прогноз
            if not df_forecast.empty and 'date' in df_forecast.columns and 'predicted_sales' in df_forecast.columns:
                dates_forecast = df_forecast['date']
                sales_forecast = df_forecast['predicted_sales']
                plt.plot(dates_forecast, sales_forecast, label='Прогноз продаж', marker='s', linestyle='-', linewidth=3, color='#F97316', markersize=8)
                
                # Добавляем доверительный интервал для прогноза
                upper_bound = sales_forecast * 1.2
                lower_bound = sales_forecast * 0.8
                plt.fill_between(dates_forecast, lower_bound, upper_bound, alpha=0.3, color='#F97316', label='Доверительный интервал (±20%)')
            
            # Настройки графика
            plt.title(f'Полный отчет: Исторические данные и прогноз продаж\nТочность модели: {model_name} - {(1-accuracy_value)*100:.1f}%', fontsize=16, fontweight='bold')
            plt.xlabel('Дата', fontsize=12)
            plt.ylabel('Продажи (руб.)', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.gcf().autofmt_xdate()
            
            # Добавляем статистику на график
            stats_text = f'История: {total_sales:,.0f} руб. | Прогноз: {total_forecast:,.0f} руб.\nСреднедневной: {avg_daily:,.0f} → {avg_forecast:,.0f} руб.'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
            
            # Сохраняем график в буфер
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            
            # Конвертируем изображение в base64
            plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            print(f"Ошибка при создании графика полного отчета: {e}")
            # Создаем простой график в случае ошибки
            plt.figure(figsize=(16, 10))
            plt.text(0.5, 0.5, 'Ошибка при создании графика', ha='center', va='center', fontsize=16)
            plt.title('Полный отчет: Исторические данные и прогноз продаж', fontsize=16)
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Создаем HTML отчет
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Full Sales Forecast Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }
                .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
                h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                .accuracy-info { background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #27ae60; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                .stat-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #dee2e6; }
                .stat-value { font-size: 20px; font-weight: bold; color: #2c3e50; }
                .stat-label { color: #6c757d; font-size: 12px; }
                .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
                .comparison-section { background-color: #f8f9fa; padding: 20px; border-radius: 8px; }
                .comparison-title { font-weight: bold; color: #2c3e50; margin-bottom: 15px; text-align: center; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; font-weight: 600; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                img { max-width: 100%; height: auto; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .footer { text-align: center; margin-top: 30px; color: #6c757d; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Полный отчет по прогнозированию продаж</h1>
                <p style="text-align: center; color: #6c757d;">Сгенерирован: {{ generation_date }}</p>
                
                <div class="accuracy-info">
                    <h3 style="margin-top: 0; color: #27ae60;">🎯 Информация о модели</h3>
                    <p><strong>Модель:</strong> {{ model_name }}</p>
                    <p><strong>Точность:</strong> {{ "%.1f"|format((1-accuracy_value)*100) }}%</p>
                    <p><strong>Дата обучения:</strong> {{ accuracy_date }}</p>
                </div>
                
                <img src="data:image/png;base64,{{ plot_base64 }}" alt="График продаж и прогноза">
                
                <div class="comparison-grid">
                    <div class="comparison-section">
                        <div class="comparison-title">📈 Исторические данные</div>
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">{{ "%.0f"|format(total_sales) }}</div>
                                <div class="stat-label">Общий объем (руб.)</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{{ "%.0f"|format(avg_daily) }}</div>
                                <div class="stat-label">Среднедневной (руб.)</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{{ "%.0f"|format(max_sales) }}</div>
                                <div class="stat-label">Максимум (руб.)</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{{ "%.0f"|format(min_sales) }}</div>
                                <div class="stat-label">Минимум (руб.)</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="comparison-section">
                        <div class="comparison-title">🔮 Прогноз на 7 дней</div>
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">{{ "%.0f"|format(total_forecast) }}</div>
                                <div class="stat-label">Общий прогноз (руб.)</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{{ "%.0f"|format(avg_forecast) }}</div>
                                <div class="stat-label">Среднедневной (руб.)</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{{ "%.0f"|format(max_forecast) }}</div>
                                <div class="stat-label">Максимум (руб.)</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{{ "%.0f"|format(min_forecast) }}</div>
                                <div class="stat-label">Минимум (руб.)</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <h2>🔮 Детали прогноза</h2>
                <table>
                    <tr>
                        <th>Дата</th>
                        <th>Прогноз продаж (руб.)</th>
                        <th>День недели</th>
                        <th>Доверительный интервал</th>
                    </tr>
                    {% for item in forecast_data %}
                    <tr>
                        <td>{{ item.date }}</td>
                        <td style="font-weight: bold; color: #2c3e50;">{{ "%.2f"|format(item.predicted_sales) }}</td>
                        <td>{{ get_day_name(item.date) }}</td>
                        <td>±20% ({{ "%.2f"|format(item.predicted_sales * 0.8) }} - {{ "%.2f"|format(item.predicted_sales * 1.2) }})</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <h2>📅 Исторические данные (последние 10 записей)</h2>
                <table>
                    <tr>
                        <th>Дата</th>
                        <th>Продажи (руб.)</th>
                        <th>День недели</th>
                        <th>Месяц</th>
                    </tr>
                    {% for item in historical_data %}
                    <tr>
                        <td>{{ item.date }}</td>
                        <td style="font-weight: bold; color: #2c3e50;">{{ "%.2f"|format(item.total_sales) }}</td>
                        <td>{{ get_day_name(item.day_of_week) }}</td>
                        <td>{{ get_month_name(item.month) }}</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <div class="footer">
                    <p>Отчет сгенерирован системой прогнозирования продаж DataPulse Analytics</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Функции для получения названий дней и месяцев
        def get_day_name(date_str_or_number):
            if isinstance(date_str_or_number, str):
                try:
                    date_obj = datetime.datetime.strptime(date_str_or_number, '%Y-%m-%d')
                    days = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
                    return days[date_obj.weekday()]
                except:
                    return 'Неизвестно'
            else:
                days = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
                return days[date_str_or_number] if 0 <= date_str_or_number < 7 else 'Неизвестно'
        
        def get_month_name(month_number):
            months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                     'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
            return months[month_number - 1] if 1 <= month_number <= 12 else 'Неизвестно'
        
        template = Template(html_template)
        html_content = template.render(
            generation_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            forecast_data=forecast_results,
            historical_data=processed_data[-10:] if processed_data else [],
            plot_base64=plot_base64,
            get_day_name=get_day_name,
            get_month_name=get_month_name,
            model_name=model_name,
            accuracy_value=accuracy_value,
            accuracy_date=accuracy_info.get('created_at', 'Неизвестно')[:10] if accuracy_info else 'Неизвестно',
            total_sales=total_sales,
            avg_daily=avg_daily,
            max_sales=max_sales,
            min_sales=min_sales,
            total_forecast=total_forecast,
            avg_forecast=avg_forecast,
            max_forecast=max_forecast,
            min_forecast=min_forecast
        )
        
        # Конвертируем HTML в PDF
        try:
            pdf_buffer = BytesIO()
            HTML(string=html_content).write_pdf(pdf_buffer)
            pdf_buffer.seek(0)
            return pdf_buffer
        except Exception as e:
            print(f"Ошибка при создании PDF для полного отчета: {e}")
            return None
        
    except Exception as e:
        print(f"Ошибка при генерации отчета: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Модуль генерации отчетов загружен")