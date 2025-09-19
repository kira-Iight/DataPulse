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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import get_engine

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

def generate_report():
    """Генерирует PDF отчет с графиками и прогнозами."""
    try:
        engine = get_engine()
        
        # 1. Получить данные из БД
        data_query = "SELECT date, total_sales FROM processed_data ORDER BY date;"
        forecast_query = "SELECT target_date, predicted_amount FROM forecast_results ORDER BY target_date;"
        
        df_data = pd.read_sql(data_query, engine)
        df_forecast = pd.read_sql(forecast_query, engine)
        
        print(f"Исторических данных: {len(df_data)} записей")
        print(f"Данных прогноза: {len(df_forecast)} записей")
        
        # Определяем пути к файлам
        plot_path = os.path.join(PROJECT_ROOT, 'plot.png')
        pdf_path = os.path.join(PROJECT_ROOT, 'sales_forecast_report.pdf')

        # 2. Создать графики
        plt.figure(figsize=(12, 6))

        # Исторические данные
        if not df_data.empty:
            dates_data = pd.to_datetime(df_data['date'])
            sales_data = df_data['total_sales']
            plt.plot(dates_data, sales_data, 
                    label='Исторические данные', marker='o', linewidth=2)
            print(f"Исторические данные: от {dates_data.min()} до {dates_data.max()}")
            print(f"Продажи: от {sales_data.min():.2f} до {sales_data.max():.2f}")

        # Прогноз
        if not df_forecast.empty:
            dates_forecast = pd.to_datetime(df_forecast['target_date'])
            sales_forecast = df_forecast['predicted_amount']
            print(f"Данные прогноза: от {dates_forecast.min()} до {dates_forecast.max()}")
            print(f"Прогноз: от {sales_forecast.min():.2f} до {sales_forecast.max():.2f}")
            
            plt.plot(dates_forecast, sales_forecast, 
                    label='Прогноз', marker='s', linestyle='--', linewidth=2, color='red')
        else:
            print("Данные прогноза пусты!")

        # Настройки графика
        plt.title('Прогноз продаж', fontsize=16)
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Продажи', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Форматирование дат
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.gcf().autofmt_xdate()
        
        # Сохранить график
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"График сохранен в: {plot_path}")

        # 3. Конвертировать изображение в base64
        with open(plot_path, "rb") as image_file:
            plot_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # 4. Создать HTML шаблон с base64 изображением
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
            
            {% if has_data %}
            <img src="data:image/png;base64,{{ plot_base64 }}" alt="График продаж и прогноза">
            
            <h2>Данные прогноза</h2>
            {% if forecast_data %}
            <table>
                <tr><th>Дата</th><th>Прогноз продаж</th></tr>
                {% for item in forecast_data %}
                <tr>
                    <td>{{ item.target_date }}</td>
                    <td>{{ "{:,.2f}".format(item.predicted_amount) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>Нет данных прогноза для отображения</p>
            {% endif %}
            
            <h2>Исторические данные (последние 10 записей)</h2>
            <table>
                <tr><th>Дата</th><th>Продажи</th></tr>
                {% for item in historical_data %}
                <tr>
                    <td>{{ item.date }}</td>
                    <td>{{ "{:,.2f}".format(item.total_sales) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>Недостаточно данных для генерации отчета</p>
            {% endif %}
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            generation_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            forecast_data=df_forecast.to_dict('records'),
            historical_data=df_data.tail(10).to_dict('records'),
            plot_base64=plot_base64,
            has_data=len(df_data) > 0
        )
        
        # 5. Конвертировать HTML в PDF
        HTML(string=html_content).write_pdf(pdf_path)
        print(f"PDF отчет сгенерирован и сохранен в: {pdf_path}")
        
        # Удаляем временный файл графика
        if os.path.exists(plot_path):
            os.remove(plot_path)
            
        return True
        
    except Exception as e:
        print(f"Ошибка при генерации отчета: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    generate_report()