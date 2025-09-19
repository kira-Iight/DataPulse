# utils/report_generator.py
import pandas as pd
from jinja2 import Template
from weasyprint import HTML
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import base64
import sys
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# utils/report_generator.py - добавим новые функции

def generate_sales_report(session_data):
    """Генерирует отчет только по историческим данным продаж."""
    try:
        processed_data = session_data.get('processed_data', [])
        
        if not processed_data:
            print("Нет исторических данных для отчета")
            return None
            
        df_data = pd.DataFrame(processed_data)
        
        # Создаем график исторических данных
        plt.figure(figsize=(12, 6))
        
        if not df_data.empty and 'date' in df_data.columns and 'total_sales' in df_data.columns:
            df_data['date'] = pd.to_datetime(df_data['date'])
            dates_data = df_data['date']
            sales_data = df_data['total_sales']
            plt.plot(dates_data, sales_data, label='Исторические данные', marker='o', linewidth=2, color='blue')
        
        # Настройки графика
        plt.title('Исторические данные продаж', fontsize=16)
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Продажи', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        # Сохраняем график в буфер
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        
        # Конвертируем изображение в base64
        plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Создаем HTML отчет
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sales Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Отчет по продажам</h1>
            <p>Сгенерирован: {{ generation_date }}</p>
            
            <img src="data:image/png;base64,{{ plot_base64 }}" alt="График продаж">
            
            <h2>Исторические данные</h2>
            <table>
                <tr>
                    <th>Дата</th>
                    <th>Продажи</th>
                    <th>День недели</th>
                    <th>Месяц</th>
                </tr>
                {% for item in historical_data %}
                <tr>
                    <td>{{ item.date }}</td>
                    <td>{{ "%.2f"|format(item.total_sales) }}</td>
                    <td>{{ item.day_of_week }}</td>
                    <td>{{ item.month }}</td>
                </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            generation_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            historical_data=processed_data,
            plot_base64=plot_base64
        )
        
        # Конвертируем HTML в PDF
        pdf_buffer = BytesIO()
        HTML(string=html_content).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        
        return pdf_buffer
        
    except Exception as e:
        print(f"Ошибка при генерации отчета по продажам: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_forecast_report(session_data):
    """Генерирует отчет только по прогнозам."""
    try:
        forecast_results = session_data.get('forecast_results', [])
        
        if not forecast_results:
            print("Нет данных прогноза для отчета")
            return None
            
        df_forecast = pd.DataFrame(forecast_results)
        
        # Создаем график прогнозов
        plt.figure(figsize=(12, 6))
        
        if not df_forecast.empty and 'date' in df_forecast.columns and 'predicted_sales' in df_forecast.columns:
            df_forecast['date'] = pd.to_datetime(df_forecast['date'])
            dates_forecast = df_forecast['date']
            sales_forecast = df_forecast['predicted_sales']
            plt.plot(dates_forecast, sales_forecast, label='Прогноз продаж', marker='s', linestyle='--', linewidth=2, color='red')
        
        # Настройки графика
        plt.title('Прогноз продаж на 7 дней', fontsize=16)
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Продажи', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        # Сохраняем график в буфер
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        
        # Конвертируем изображение в base64
        plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Создаем HTML отчет
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forecast Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Отчет по прогнозированию продаж</h1>
            <p>Сгенерирован: {{ generation_date }}</p>
            
            <img src="data:image/png;base64,{{ plot_base64 }}" alt="График прогноза продаж">
            
            <h2>Данные прогноза</h2>
            <table>
                <tr>
                    <th>Дата</th>
                    <th>Прогноз продаж</th>
                    <th>День недели</th>
                </tr>
                {% for item in forecast_data %}
                <tr>
                    <td>{{ item.date }}</td>
                    <td>{{ "%.2f"|format(item.predicted_sales) }}</td>
                    <td>{{ get_day_name(item.date) }}</td>
                </tr>
                {% endfor %}
            </table>
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
            get_day_name=get_day_name
        )
        
        # Конвертируем HTML в PDF
        pdf_buffer = BytesIO()
        HTML(string=html_content).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        
        return pdf_buffer
        
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
        
        df_data = pd.DataFrame(processed_data)
        df_forecast = pd.DataFrame(forecast_results)
        
        # Создаем график
        plt.figure(figsize=(12, 6))
        
        # Исторические данные
        if not df_data.empty and 'date' in df_data.columns and 'total_sales' in df_data.columns:
            df_data['date'] = pd.to_datetime(df_data['date'])
            dates_data = df_data['date']
            sales_data = df_data['total_sales']
            plt.plot(dates_data, sales_data, label='Исторические данные', marker='o', linewidth=2)
        
        # Прогноз
        if not df_forecast.empty and 'date' in df_forecast.columns and 'predicted_sales' in df_forecast.columns:
            df_forecast['date'] = pd.to_datetime(df_forecast['date'])
            dates_forecast = df_forecast['date']
            sales_forecast = df_forecast['predicted_sales']
            plt.plot(dates_forecast, sales_forecast, label='Прогноз', marker='s', linestyle='--', linewidth=2, color='red')
        
        # Настройки графика
        plt.title('Прогноз продаж', fontsize=16)
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Продажи', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        # Сохраняем график в буфер
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        
        # Конвертируем изображение в base64
        plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Создаем HTML отчет
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sales Forecast Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Отчет по прогнозированию продаж</h1>
            <p>Сгенерирован: {{ generation_date }}</p>
            
            <img src="data:image/png;base64,{{ plot_base64 }}" alt="График продаж и прогноза">
            
            <h2>Данные прогноза</h2>
            <table>
                <tr><th>Дата</th><th>Прогноз продаж</th></tr>
                {% for item in forecast_data %}
                <tr>
                    <td>{{ item.date }}</td>
                    <td>{{ "%.2f"|format(item.predicted_sales) }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Исторические данные (последние 10 записей)</h2>
            <table>
                <tr><th>Дата</th><th>Продажи</th></tr>
                {% for item in historical_data %}
                <tr>
                    <td>{{ item.date }}</td>
                    <td>{{ "%.2f"|format(item.total_sales) }}</td>
                </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            generation_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            forecast_data=forecast_results,
            historical_data=processed_data[-10:] if processed_data else [],
            plot_base64=plot_base64
        )
        
        # Конвертируем HTML в PDF
        pdf_buffer = BytesIO()
        HTML(string=html_content).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        
        return pdf_buffer
        
    except Exception as e:
        print(f"Ошибка при генерации отчета: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    generate_full_report()