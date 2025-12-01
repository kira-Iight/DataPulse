import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
from datetime import datetime
import threading
import logging
import numpy as np

# Импорт модулей приложения
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import AppConfig, DataValidationRules
from data_manager import DataManager
from ml_engine import RidgeRegressionEngine
from logging_config import setup_logging
from report_generator import ReportGenerator

# Инициализация системы логирования
setup_logging()

class ModernTheme:
    """Класс для хранения цветовой схемы и шрифтов приложения"""
    # Использование конфигурации из AppConfig
    COLORS = AppConfig.COLORS
    FONTS = AppConfig.FONTS

class SalesForecastApp:
    """Основной класс приложения для прогнозирования продаж"""
    
    def __init__(self, root):
        # Инициализация главного окна
        self.root = root
        self.root.title("DataPulse")
        self.root.geometry("1400x900")
        
        # Установка фонового цвета
        self.root.configure(bg=ModernTheme.COLORS['background'])
        
        # Инициализация менеджеров данных
        self.data_manager = DataManager()
        self.forecast_engine = RidgeRegressionEngine()
        self.report_generator = ReportGenerator()
        self.logger = logging.getLogger(__name__)
        
        # Попытка установки иконки приложения
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass  # Если иконка не найдена, продолжаем без нее
        
        # Настройка стилей и создание интерфейса
        self.setup_styles()
        
        # Инициализация переменных для хранения данных
        self.raw_data = None          # Исходные данные из CSV
        self.processed_data = None    # Обработанные данные
        self.forecast_results = None  # Результаты прогнозирования
        self.model_accuracy = []      # Метрики точности модели
        self.model_comparison_results = None  # Результаты сравнения моделей
        
        # Создание элементов интерфейса
        self.create_widgets()
                
    def setup_styles(self):
        """Настройка визуальных стилей для элементов интерфейса"""
        style = ttk.Style()
        style.theme_use('clam')  # Использование современной темы
        
        # Настройка стилей для фреймов
        style.configure('TFrame', background=ModernTheme.COLORS['background'])
        style.configure('TLabel', font=ModernTheme.FONTS['normal'], background=ModernTheme.COLORS['card'])
        style.configure('Title.TLabel', font=ModernTheme.FONTS['title'], background=ModernTheme.COLORS['background'])
        style.configure('T.TLabel', font=ModernTheme.FONTS['title'], background=ModernTheme.COLORS['light'], foreground=ModernTheme.COLORS['dark']) 
        
        # Настройка стилей для кнопок
        style.configure('TButton', font=ModernTheme.FONTS['normal'], padding=6)
        style.configure('Primary.TButton', background=ModernTheme.COLORS['primary'], foreground='white')
        style.configure('Secondary.TButton', background=ModernTheme.COLORS['secondary'], foreground='white')
        style.configure('Success.TButton', background=ModernTheme.COLORS['success'], foreground='white')
        style.configure('Warning.TButton', background=ModernTheme.COLORS['warning'], foreground='white')
        
        # Стиль для карточек (блоков с контентом)
        style.configure('Card.TFrame', background=ModernTheme.COLORS['card'], relief='raised', borderwidth=1)
        
        # Стили для таблицы Treeview
        style.configure('Treeview', 
                       background=ModernTheme.COLORS['card'],
                       foreground=ModernTheme.COLORS['dark'],
                       fieldbackground=ModernTheme.COLORS['card'],
                       borderwidth=0,
                       font=ModernTheme.FONTS['small'])
        
        style.configure('Treeview.Heading', 
                       background=ModernTheme.COLORS['primary'],
                       foreground='white',
                       padding=8,
                       font=ModernTheme.FONTS['small'])
        
        # Эффекты при наведении на заголовки таблицы
        style.map('Treeview.Heading', 
                 background=[('active', ModernTheme.COLORS['primary_light'])])
        
        # Стили для вкладок (Notebook)
        style.configure('TNotebook', background=ModernTheme.COLORS['background'], borderwidth=0)
        style.configure('TNotebook.Tab', 
                       background=ModernTheme.COLORS['secondary'],
                       foreground='white',
                       padding=[15, 5],
                       font=ModernTheme.FONTS['normal'])
        
        # Эффекты для вкладок при выборе и наведении
        style.map('TNotebook.Tab', 
                 background=[('selected', ModernTheme.COLORS['primary']),
                           ('active', ModernTheme.COLORS['primary_light'])])
        
    def create_widgets(self):
        """Создание всех элементов пользовательского интерфейса"""
        # Главный контейнер
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Заголовок приложения
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        # Название приложения
        title_label = ttk.Label(header_frame, 
                 text="DataPulse",
                 style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Подзаголовок (пока пустой, можно заполнить позже)
        subtitle_label = ttk.Label(header_frame, 
                 font=ModernTheme.FONTS['subtitle'])
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))

        # Основная область контента
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Боковая панель с кнопками
        sidebar_frame = ttk.Frame(content_frame, width=280, style='Card.TFrame')
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        sidebar_frame.pack_propagate(False)  # Фиксированная ширина
        
        # Основная рабочая область
        main_area_frame = ttk.Frame(content_frame)
        main_area_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Заполнение боковой панели и основной области
        self.create_sidebar(sidebar_frame)
        self.create_main_area(main_area_frame)
    
    def get_chart_data(self, historical_days=30):
        """Подготовка данных для построения графиков"""
        if self.processed_data is None or self.processed_data.empty:
            return pd.DataFrame(), []  # Пустые данные если нет обработанных данных
        
        # Ограничение исторических данных последними N днями
        df_hist = self.processed_data.copy()
        if 'date' in df_hist.columns:
            df_hist = df_hist.sort_values('date')
            limited_historical = df_hist.tail(historical_days)
        else:
            limited_historical = df_hist
        
        # Возврат ограниченных исторических данных и прогнозов
        return limited_historical, self.forecast_results or []

    def update_forecast_chart(self):
        """Обновление графика прогноза с доверительными интервалами"""
        if self.forecast_results:
            self.ax.clear()  # Очистка предыдущего графика
            
            # Получение данных для графика
            historical_data, forecasts = self.get_chart_data(historical_days=30)
            
            # Отображение исторических данных
            if historical_data is not None and not historical_data.empty:
                dates = historical_data['date']
                sales = historical_data['total_sales']
                self.ax.plot(dates, sales, 
                        label='Исторические данные (30 дней)', 
                        marker='o', 
                        linewidth=2.5, 
                        color=ModernTheme.COLORS['primary'],
                        markersize=4,
                        alpha=0.8)
            
            # Отображение прогноза
            if forecasts:
                forecast_dates = [pd.to_datetime(pred['date']) for pred in forecasts]
                forecast_sales = [pred['predicted_sales'] for pred in forecasts]
                
                self.ax.plot(forecast_dates, forecast_sales, 
                        label='Прогноз', 
                        marker='s', 
                        linewidth=3, 
                        color=ModernTheme.COLORS['success'],
                        markersize=6)
                
                # Добавление доверительных интервалов если они есть
                if 'confidence_interval' in forecasts[0]:
                    upper_bound = [pred['confidence_interval']['upper'] for pred in forecasts]
                    lower_bound = [pred['confidence_interval']['lower'] for pred in forecasts]
                    
                    uncertainty_pct = forecasts[0]['confidence_interval']['uncertainty_pct']
                    confidence_level = forecasts[0]['confidence_interval']['confidence_level']
                    
                    # Заливка области доверительного интервала
                    self.ax.fill_between(forecast_dates, lower_bound, upper_bound, 
                                    alpha=0.2, 
                                    color=ModernTheme.COLORS['success'],
                                    label=f'Доверительный интервал ±{uncertainty_pct:.1f}%')
                
                # Расчет общей суммы прогноза
                total_forecast = sum(pred['predicted_sales'] for pred in forecasts)
                
                # Обновление информации о прогнозе в интерфейсе
                if forecasts and 'confidence_interval' in forecasts[0]:
                    avg_uncertainty = np.mean([pred['confidence_interval']['uncertainty_pct'] for pred in forecasts])
                    self.forecast_info_label.config(
                        text=f"Прогноз: {total_forecast:,.0f} руб. | Неопределенность: ±{avg_uncertainty:.1f}%"
                    )
                else:
                    self.forecast_info_label.config(
                        text=f"Прогноз: {total_forecast:,.0f} руб."
                    )
            
            # Настройка оформления графика
            self.ax.set_title(f"Прогноз продаж на 7 дней с доверительными интервалами", 
                            fontsize=14, fontweight='bold', pad=20)
            self.ax.set_xlabel("Дата", fontsize=12)
            self.ax.set_ylabel("Продажи (руб.)", fontsize=12)
            self.ax.legend()  # Легенда графика
            self.ax.grid(True, alpha=0.3)  # Сетка
            
            # Поворот подписей дат для лучшей читаемости
            plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
            self.canvas.draw()  # Обновление отображения графика

    def update_stats_chart(self):
        """Обновление графика статистики (ограничено 30 днями)"""
        if self.processed_data is not None and not self.processed_data.empty:
            self.stats_ax.clear()  # Очистка предыдущего графика

            # Получение исторических данных
            historical_data, _ = self.get_chart_data(historical_days=30)
            
            if not historical_data.empty:
                dates = historical_data['date']
                sales = historical_data['total_sales']
                
                # Построение графика фактических данных
                self.stats_ax.plot(dates, sales, marker='o', linewidth=2.5, 
                                color=ModernTheme.COLORS['primary'], markersize=4, alpha=0.8)
                
                # Добавление скользящего среднего если достаточно данных
                if len(sales) >= 7:
                    rolling_mean = sales.rolling(window=7).mean()
                    self.stats_ax.plot(dates, rolling_mean, linewidth=2, 
                                    color=ModernTheme.COLORS['warning'], linestyle='--', alpha=0.7)
                
                # Настройка оформления графика
                self.stats_ax.set_title("Динамика продаж (последние 30 дней)", fontsize=14, fontweight='bold', pad=20)
                self.stats_ax.set_xlabel("Дата", fontsize=12)
                self.stats_ax.set_ylabel("Продажи (руб.)", fontsize=12)
                self.stats_ax.grid(True, alpha=0.3)
                self.stats_ax.legend(['Фактические данные', 'Скользящее среднее (7 дней)'])
                
                # Поворот подписей дат
                plt.setp(self.stats_ax.xaxis.get_majorticklabels(), rotation=45)
                self.stats_canvas.draw()  # Обновление отображения

    def create_sidebar(self, parent):
        """Создание боковой панели с кнопками управления"""
        
        # Секция управления данными
        data_label = ttk.Label(parent, 
                 text="Данные", 
                 font=ModernTheme.FONTS['normal'])
        data_label.pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        # Кнопка загрузки CSV файла
        load_button = ttk.Button(parent, 
                  text="Загрузить CSV", 
                  command=self.load_csv_file,
                  style='Primary.TButton')
        load_button.pack(fill=tk.X, padx=20, pady=5)
        
        # Кнопка очистки данных
        clear_button = ttk.Button(parent, 
                  text="Очистить данные", 
                  command=self.clear_data,
                  style='Secondary.TButton')
        clear_button.pack(fill=tk.X, padx=20, pady=5)
        
        # Секция прогнозирования
        forecast_label = ttk.Label(parent, 
                 text="Прогнозирование", 
                 font=ModernTheme.FONTS['normal'])
        forecast_label.pack(anchor=tk.W, padx=20, pady=(20, 5))
        
        # Кнопка обучения модели и прогнозирования
        train_button = ttk.Button(parent, 
                  text="Обучить и спрогнозировать", 
                  command=self.run_forecast,
                  style='Success.TButton')
        train_button.pack(fill=tk.X, padx=20, pady=5)
        
        # Секция отчетов
        reports_label = ttk.Label(parent, 
                 text="Отчеты", 
                 font=ModernTheme.FONTS['normal'])
        reports_label.pack(anchor=tk.W, padx=20, pady=(20, 5))
        
        # Кнопки генерации различных отчетов
        sales_report_button = ttk.Button(parent, 
                  text="Отчет по продажам", 
                  command=self.generate_sales_report)
        sales_report_button.pack(fill=tk.X, padx=20, pady=2)
        
        forecast_report_button = ttk.Button(parent, 
                  text="Отчет по прогнозам", 
                  command=self.generate_forecast_report)
        forecast_report_button.pack(fill=tk.X, padx=20, pady=2)
        
        full_report_button = ttk.Button(parent, 
                  text="Полный отчет", 
                  command=self.generate_full_report)
        full_report_button.pack(fill=tk.X, padx=20, pady=2)
        
    def create_main_area(self, parent):
        """Создание основной рабочей области с вкладками"""
        # Создание панели вкладок
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Создание вкладок
        self.data_frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.data_frame, text="Данные")
        
        self.forecast_frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.forecast_frame, text="Прогноз")
        
        self.stats_frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.stats_frame, text="Статистика")
        
        # Заполнение вкладок содержимым
        self.create_data_tab()
        self.create_forecast_tab()
        self.create_stats_tab()
        
    def create_data_tab(self):
        """Создание вкладки с таблицей данных"""
        # Заголовок вкладки
        header = ttk.Frame(self.data_frame)
        header.pack(fill=tk.X, pady=(0, 15))
        
        data_title = ttk.Label(header, 
                 text="Исторические данные продаж", 
                 style='Title.TLabel')
        data_title.pack(side=tk.LEFT)
        
        # Метка для отображения информации о данных
        self.data_info_label = ttk.Label(header, 
                                        text="Данные не загружены",
                                        style='Title.TLabel')
        self.data_info_label.pack(side=tk.RIGHT)
        
        # Контейнер для таблицы
        card_frame = ttk.Frame(self.data_frame, style='Card.TFrame')
        card_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создание таблицы Treeview
        columns = ('Дата', 'Продажи', 'День недели', 'Месяц', 'Квартал', 'Выходной', 'Праздник')
        self.data_tree = ttk.Treeview(card_frame, columns=columns, show='headings', height=20)
        
        # Настройка ширины колонок
        column_widths = {'Дата': 100, 'Продажи': 120, 'День недели': 90, 
                        'Месяц': 70, 'Квартал': 70, 'Выходной': 70, 'Праздник': 70}
        
        # Установка заголовков и ширины колонок
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=column_widths.get(col, 100), anchor=tk.CENTER)
        
        # Добавление вертикальной полосы прокрутки
        scrollbar = ttk.Scrollbar(card_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # Размещение таблицы и полосы прокрутки
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
    def create_forecast_tab(self):
        """Создание вкладки с графиком прогноза"""
        # Заголовок вкладки
        header = ttk.Frame(self.forecast_frame)
        header.pack(fill=tk.X, pady=(0, 15))
        
        forecast_title = ttk.Label(header, 
                 text="Прогноз продаж на 7 дней", 
                 style='Title.TLabel')
        forecast_title.pack(side=tk.LEFT)
        
        # Метка для отображения информации о прогнозе
        self.forecast_info_label = ttk.Label(header, 
                                           text="Прогноз не выполнен", 
                                           style='Title.TLabel')
        self.forecast_info_label.pack(side=tk.RIGHT)
        
        # Контейнер для графика
        card_frame = ttk.Frame(self.forecast_frame, style='Card.TFrame')
        card_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создание фигуры matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.ax.set_title("Прогноз продаж", fontsize=14, fontweight='bold', pad=20)
        self.ax.set_xlabel("Дата", fontsize=12)
        self.ax.set_ylabel("Продажи (руб.)", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # Настройка цветов графика
        self.fig.patch.set_facecolor(ModernTheme.COLORS['card'])
        self.ax.set_facecolor(ModernTheme.COLORS['card'])
        
        # Встраивание графика в tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, card_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_stats_tab(self):
        """Создание вкладки со статистикой"""
        # Заголовок вкладки
        stats_title = ttk.Label(self.stats_frame, 
                 text="Статистика данных", 
                 style='Title.TLabel')
        stats_title.pack(anchor=tk.W, pady=(0, 15))
        
        # Фрейм для метрик
        metrics_frame = ttk.Frame(self.stats_frame)
        metrics_frame.pack(fill=tk.X, pady=(0, 20))

        # Настройка равномерного распределения колонок
        for i in range(8):
            metrics_frame.columnconfigure(i, weight=1)

        # Словарь для хранения меток метрик
        self.metrics = {}
        
        # Данные для метрик: (текст, ключ, значение по умолчанию, цвет, строка, колонка)
        metrics_data = [
            ("Записей", "total_records", "0", ModernTheme.COLORS['primary'], 0, 0),
            ("Общий объем", "total_sales", "0 руб.", ModernTheme.COLORS['success'], 0, 1),
            ("Среднедневной", "avg_daily", "0 руб.", ModernTheme.COLORS['warning'], 0, 2),
            ("MAE", "model_mae", "N/A", ModernTheme.COLORS['primary'], 0, 3),
            ("Период", "date_range", "N/A", ModernTheme.COLORS['success'], 0, 4),
            ("Максимум", "max_sales", "0 руб.", ModernTheme.COLORS['danger'], 0, 5),
            ("Минимум", "min_sales", "0 руб.", ModernTheme.COLORS['secondary'], 0, 6),
            ("RMSE", "model_rmse", "N/A", ModernTheme.COLORS['warning'], 0, 7)
        ]

        # Создание карточек с метриками
        for label, key, default, color, row, col in metrics_data:
            metric_card = ttk.Frame(metrics_frame, style='Card.TFrame', height=100)
            metric_card.grid(row=row, column=col, padx=5, pady=10, sticky="nsew")
            
            # Метка названия метрики
            label_widget = ttk.Label(metric_card, 
                    text=label, 
                    font=ModernTheme.FONTS['small'])
            label_widget.pack(pady=(15, 5))
            
            # Метка значения метрики
            value_label = ttk.Label(metric_card, 
                                text=default, 
                                font=ModernTheme.FONTS['metric'])
            value_label.pack(pady=(0, 15))
            
            # Сохранение ссылки на метку значения
            self.metrics[key] = value_label
        
        # Контейнер для графика статистики
        card_frame = ttk.Frame(self.stats_frame, style='Card.TFrame')
        card_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создание графика статистики
        self.stats_fig, self.stats_ax = plt.subplots(figsize=(12, 6))
        self.stats_ax.set_title("Динамика продаж", fontsize=14, fontweight='bold', pad=20)
        self.stats_ax.set_xlabel("Дата", fontsize=12)
        self.stats_ax.set_ylabel("Продажи (руб.)", fontsize=12)
        self.stats_ax.grid(True, alpha=0.3)
        
        # Настройка цветов графика
        self.stats_fig.patch.set_facecolor(ModernTheme.COLORS['card'])
        self.stats_ax.set_facecolor(ModernTheme.COLORS['card'])
        
        # Встраивание графика в tkinter
        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, card_frame)
        self.stats_canvas.draw()
        self.stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def load_csv_file(self):
        """Загрузка CSV файла с данными продаж"""
        # Открытие диалога выбора файла
        file_path = filedialog.askopenfilename(
            title="Выберите CSV файл",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:                
                # Загрузка и обработка данных через DataManager
                self.raw_data = self.data_manager.load_data_from_csv(file_path)
                self.processed_data = self.data_manager.preprocess_data(self.raw_data)
                
                # Передача пути к файлу в модель для прогнозирования
                self.forecast_engine.set_current_file_path(file_path)
                
                # Обновление интерфейса
                self.update_data_table()
                self.update_stats()
                self.update_stats_chart()
                
                # Сообщение об успешной загрузке
                success_msg = f"Файл загружен успешно! Обработано {len(self.processed_data)} записей"
                messagebox.showinfo("Успех", success_msg)
                
            except Exception as e:
                # Обработка ошибок при загрузке
                error_msg = f"Ошибка при загрузке файла: {str(e)}"
                self.logger.error(error_msg)
                messagebox.showerror("Ошибка", error_msg)
    
    def update_data_table(self):
        """Обновление таблицы с данными продаж"""
        # Очистка текущего содержимого таблицы
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Проверка наличия данных для отображения
        if self.processed_data is not None and not self.processed_data.empty:
            # Получение статистики по данным
            stats = self.data_manager.get_data_statistics(self.processed_data)
            
            # Обновление информационной метки
            info_text = f"Записей: {stats['total_records']} | Период: {stats['date_range']['start']} - {stats['date_range']['end']}"
            self.data_info_label.config(text=info_text)
            
            # Заполнение таблицы данными
            for _, row in self.processed_data.iterrows():
                # Преобразование номера дня недели в название
                day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
                day_name = day_names[int(row['day_of_week'])] if 'day_of_week' in row else 'Н/Д'
                
                # Преобразование булевых значений в Да/Нет
                is_weekend = 'Да' if row.get('is_weekend', False) else 'Нет'
                is_holiday = 'Да' if row.get('is_holiday', False) else 'Нет'
                
                # Добавление строки в таблицу
                self.data_tree.insert('', 'end', values=(
                    str(row['date'].date()),  # Дата
                    f"{row['total_sales']:,.2f}",  # Продажи с форматированием
                    day_name,  # День недели
                    int(row['month']),  # Месяц
                    f"Q{int(row['quarter'])}",  # Квартал
                    is_weekend,  # Выходной день
                    is_holiday  # Праздничный день
                ))
        else:
            # Если данных нет, отображаем соответствующее сообщение
            self.data_info_label.config(text="Данные не загружены")
    
    def _calculate_simple_confidence_interval(self, predictions, historical_data, confidence_level=0.95):
        """Расчет простых доверительных интервалов для прогнозов"""
        try:
            confidence_intervals = []
            
            # Простой расчет: фиксированный процент от прогнозируемого значения
            for pred in predictions:
                predicted_value = pred['predicted_sales']
                uncertainty_pct = 15.0  # Фиксированный процент неопределенности
                
                # Расчет границ интервала
                margin = (predicted_value * uncertainty_pct) / 100
                
                # Добавление интервала в список
                confidence_intervals.append({
                    'lower': max(predicted_value - margin, 0),  # Нижняя граница (не меньше 0)
                    'upper': predicted_value + margin,           # Верхняя граница
                    'uncertainty_pct': uncertainty_pct,          # Процент неопределенности
                    'confidence_level': confidence_level         # Уровень доверия
                })
            
            return confidence_intervals
        
        except Exception as e:
            # Логирование ошибки при расчете интервалов
            self.logger.error(f"Ошибка расчета доверительных интервалов: {str(e)}")
            return []

    def update_stats(self):
        """Обновление статистических метрик на вкладке статистики"""
        if self.processed_data is not None and not self.processed_data.empty:
            # Получение статистики по данным
            stats = self.data_manager.get_data_statistics(self.processed_data)
            
            # Обновление значений метрик в интерфейсе
            self.metrics['total_records'].config(text=str(stats['total_records']))
            self.metrics['total_sales'].config(text=f"{stats['total_sales']:,.2f} руб.")
            self.metrics['avg_daily'].config(text=f"{stats['avg_daily']:,.2f} руб.")
            self.metrics['max_sales'].config(text=f"{stats['max_sales']:,.2f} руб.")
            self.metrics['min_sales'].config(text=f"{stats['min_sales']:,.2f} руб.")
            self.metrics['date_range'].config(text=f"{stats['date_range']['days']} дней")
            
            # Обновление метрик модели если они доступны
            if self.model_accuracy:
                self.update_model_metrics()

    def run_forecast(self):
        """Запуск процесса прогнозирования продаж"""
        # Проверка наличия данных
        if self.processed_data is None or self.processed_data.empty:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        
        # Проверка минимального количества данных
        if len(self.processed_data) < AppConfig.MIN_DATA_POINTS:
            messagebox.showwarning("Предупреждение", 
                                f"Недостаточно данных для прогнозирования!\nНужно минимум {AppConfig.MIN_DATA_POINTS} записей.")
            return
        
        # Создание окна прогресса
        progress = tk.Toplevel(self.root)
        progress.title("Анализ данных и прогноз")
        progress.geometry("400x150")
        progress.configure(bg=ModernTheme.COLORS['background'])
        progress.transient(self.root)  # Сделать окно модальным
        progress.grab_set()  # Захват фокуса
        
        # Центрирование окна прогресса
        progress.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - progress.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - progress.winfo_height()) // 2
        progress.geometry(f"+{x}+{y}")
        
        # Текст прогресса
        ttk.Label(progress, 
                text="Анализ исторических данных...", 
                font=ModernTheme.FONTS['normal'],
                background=ModernTheme.COLORS['background']).pack(pady=20)
        
        # Полоса прогресса
        progress_bar = ttk.Progressbar(progress, mode='indeterminate', length=300)
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_bar.start()  # Запуск анимации прогресса
                
        def train_and_predict():
            """Функция для обучения модели и прогнозирования в отдельном потоке"""
            try:
                # Подготовка данных для сессии
                session_data = {
                    'processed_data': self.processed_data.to_dict('records'),
                    'model_accuracy': self.model_accuracy
                }
                
                # Логирование начала процесса
                self.logger.info("=" * 50)
                self.logger.info("НАЧАЛО ПРОГНОЗИРОВАНИЯ")
                self.logger.info(f"Размер обучающих данных: {len(self.processed_data)}")
                self.logger.info(f"Диапазон дат: {self.processed_data['date'].min()} - {self.processed_data['date'].max()}")
                self.logger.info(f"Диапазон продаж: {self.processed_data['total_sales'].min():.0f} - {self.processed_data['total_sales'].max():.0f}")
                self.logger.info(f"Средние продажи: {self.processed_data['total_sales'].mean():.0f}")
                self.logger.info(f"Медиана продаж: {self.processed_data['total_sales'].median():.0f}")
                
                # Анализ последних 7 дней
                last_7_days = self.processed_data.tail(7)
                if not last_7_days.empty:
                    self.logger.info(f"Последние 7 дней: {last_7_days['total_sales'].mean():.0f} в среднем")
                    self.logger.info(f"Тренд последних 7 дней: {self._calculate_trend(last_7_days['total_sales']):.2f}%")
                
                # Конвертация дат в строки для сериализации
                for item in session_data['processed_data']:
                    if hasattr(item['date'], 'strftime'):
                        item['date'] = item['date'].strftime('%Y-%m-%d')
                
                # Обучение модели
                model, accuracy = self.forecast_engine.train_model(session_data, optimize_hyperparams=False)
                
                # Проверка успешности обучения
                if model is None:
                    self.root.after(0, progress.destroy)
                    self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не удалось обучить модель"))
                    return
                
                # Прогнозирование на 7 дней
                predictions = self.forecast_engine.make_predictions(model, session_data, days_to_forecast=7)

                # Проверка успешности прогнозирования
                if not predictions:
                    self.root.after(0, progress.destroy)
                    self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не удалось создать прогноз"))
                    return
                
                # Получение метрик модели
                model_metrics = self.forecast_engine.get_model_metrics(session_data)
                
                # Сохранение информации о модели
                model_info = {
                    'model_name': model_metrics['model_name'],
                    'accuracy': accuracy,
                    'mae': model_metrics['mae'],
                    'rmse': model_metrics['rmse'],
                    'mae_absolute': model_metrics['mae_absolute'],
                    'rmse_absolute': model_metrics['rmse_absolute'],
                    'features_used': model_metrics['features_used'],
                    'training_size': model_metrics['training_size'],
                    'created_at': model_metrics['created_at']
                }
                
                # Добавление информации о модели в список
                if not hasattr(self, 'model_accuracy'):
                    self.model_accuracy = []
                self.model_accuracy.append(model_info)

                # Логирование результатов прогноза
                self.logger.info("РЕЗУЛЬТАТЫ ПРОГНОЗА:")
                pred_values = [p['predicted_sales'] for p in predictions]
                self.logger.info(f"Прогнозируемые значения: {[f'{v:.0f}' for v in pred_values]}")
                self.logger.info(f"Средний прогноз: {np.mean(pred_values):.0f}")
                self.logger.info(f"Диапазон прогноза: {min(pred_values):.0f} - {max(pred_values):.0f}")

                # Детальный анализ согласованности прогноза с историей
                if predictions and not self.processed_data.empty:
                    historical_30_days = self.processed_data.tail(30)
                    hist_stats = {
                        'mean': historical_30_days['total_sales'].mean(),
                        'median': historical_30_days['total_sales'].median(),
                        'std': historical_30_days['total_sales'].std(),
                        'min': historical_30_days['total_sales'].min(),
                        'max': historical_30_days['total_sales'].max()
                    }
                    
                    pred_stats = {
                        'mean': np.mean(pred_values),
                        'median': np.median(pred_values),
                        'std': np.std(pred_values),
                        'min': min(pred_values),
                        'max': max(pred_values)
                    }
                    
                    self.logger.info("ДЕТАЛЬНЫЙ АНАЛИЗ СОГЛАСОВАННОСТИ:")
                    self.logger.info(f"История (30 дней): ср={hist_stats['mean']:.0f}, мед={hist_stats['median']:.0f}")
                    self.logger.info(f"Прогноз (7 дней):  ср={pred_stats['mean']:.0f}, мед={pred_stats['median']:.0f}")
                    self.logger.info(f"Отклонение среднего: {((pred_stats['mean'] - hist_stats['mean']) / hist_stats['mean']) * 100:+.1f}%")
                    self.logger.info(f"Отклонение медианы: {((pred_stats['median'] - hist_stats['median']) / hist_stats['median']) * 100:+.1f}%")

                # Добавление доверительных интервалов к прогнозам
                if predictions and not self.processed_data.empty:
                    historical_sales = self.processed_data['total_sales'].values
                    confidence_intervals = self._calculate_simple_confidence_interval(
                        predictions, historical_sales
                    )
                    
                    # Объединение прогнозов с доверительными интервалами
                    for i, pred in enumerate(predictions):
                        pred['confidence_interval'] = confidence_intervals[i]
                    
                    self.logger.info(f"Добавлены доверительные интервалы: ±{confidence_intervals[0]['uncertainty_pct']}%")
                
                # Сравнение прогноза с историческими данными
                last_30_days = self.processed_data['total_sales'].tail(30)
                hist_mean = last_30_days.mean()
                pred_mean = np.mean(pred_values)
                deviation = ((pred_mean - hist_mean) / hist_mean) * 100
                
                self.logger.info(f"Отклонение от исторического среднего: {deviation:+.1f}%")
                
                # Обновление данных приложения
                self.forecast_results = predictions
                self.model_accuracy = session_data.get('model_accuracy', [])
                
                # Обновление интерфейса в основном потоке
                self.root.after(0, self.update_forecast_chart)
                self.root.after(0, self.update_model_metrics)
                self.root.after(0, progress.destroy)
                
                # Получение метрик для отображения в сообщении
                metrics = self.forecast_engine.get_model_metrics(session_data)
            
                # Формирование сообщения об успехе
                success_msg = "Прогноз выполнен успешно!"
                    
                self.root.after(0, lambda: messagebox.showinfo("Успех", success_msg))
                
                # Логирование успешного завершения
                self.logger.info("ПРОГНОЗИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО")
                self.logger.info("=" * 50)
                    
            except Exception as e:
                # Обработка ошибок при прогнозировании
                self.root.after(0, progress.destroy)
                error_msg = f"Ошибка при прогнозировании: {str(e)}"
                self.logger.error(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))
        
        # Запуск процесса прогнозирования в отдельном потоке
        thread = threading.Thread(target=train_and_predict)
        thread.daemon = True
        thread.start()

    def _calculate_trend(self, sales_data):
        """Вычисление процентного тренда данных продаж
        
        Args:
            sales_data: Series с данными продаж
        
        Returns:
            Процент изменения от начала к концу периода
        """
        try:
            # Для расчета тренда нужно минимум 2 точки
            if len(sales_data) < 2:
                return 0.0
            
            # Расчет процентного изменения
            first_value = sales_data.iloc[0]
            last_value = sales_data.iloc[-1]
            
            if first_value > 0:
                return ((last_value - first_value) / first_value) * 100
            
            return 0.0
        except:
            # В случае ошибки возвращаем 0
            return 0.0

    def update_model_metrics(self):
        """Обновление метрик модели (MAE и RMSE) в интерфейсе"""
        if self.model_accuracy:
            # Используем последнюю обученную модель
            latest_model = self.model_accuracy[-1]
            
            # Обновление MAE (Mean Absolute Error)
            mae = latest_model.get('mae_absolute', 0)
            if mae > 0:
                mae_text = f"{mae:.0f} руб."
            else:
                mae_text = "N/A"
            self.metrics['model_mae'].config(text=mae_text)
            
            # Обновление RMSE (Root Mean Square Error)
            rmse = latest_model.get('rmse_absolute', 0)
            if rmse > 0:
                rmse_text = f"{rmse:.0f} руб."
            else:
                rmse_text = "N/A"
            self.metrics['model_rmse'].config(text=rmse_text)

    def clear_data(self):
        """Очистка всех данных и сброс интерфейса"""
        # Подтверждение действия пользователем
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите очистить все данные?"):
            # Сброс всех переменных с данными
            self.raw_data = None
            self.processed_data = None
            self.forecast_results = None
            self.model_accuracy = []
            self.model_comparison_results = None
            
            # Очистка таблицы данных
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Сброс графика прогноза
            self.ax.clear()
            self.ax.set_title("Прогноз продаж", fontsize=14, fontweight='bold', pad=20)
            self.ax.set_xlabel("Дата", fontsize=12)
            self.ax.set_ylabel("Продажи (руб.)", fontsize=12)
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
            
            # Сброс графика статистики
            self.stats_ax.clear()
            self.stats_ax.set_title("Динамика продаж", fontsize=14, fontweight='bold', pad=20)
            self.stats_ax.set_xlabel("Дата", fontsize=12)
            self.stats_ax.set_ylabel("Продажи (руб.)", fontsize=12)
            self.stats_ax.grid(True, alpha=0.3)
            self.stats_canvas.draw()
            
            # Сброс значений метрик
            self.metrics['total_records'].config(text="0")
            self.metrics['total_sales'].config(text="0 руб.")
            self.metrics['avg_daily'].config(text="0 руб.")
            self.metrics['max_sales'].config(text="0 руб.")
            self.metrics['min_sales'].config(text="0 руб.")
            self.metrics['model_mae'].config(text="N/A") 
            self.metrics['model_rmse'].config(text="N/A")  
            self.metrics['date_range'].config(text="N/A")
            
            # Сброс информационных меток
            self.data_info_label.config(text="Данные не загружены")
            self.forecast_info_label.config(text="Прогноз не выполнен")
            
            # Сообщение об успешной очистке
            messagebox.showinfo("Успех", "Данные очищены")

    def generate_sales_report(self):
        """Генерация PDF отчета по историческим данным продаж"""
        # Проверка наличия данных
        if self.processed_data is None or self.processed_data.empty:
            messagebox.showwarning("Предупреждение", "Нет данных для генерации отчета!")
            return
        
        try:
            # Подготовка данных для отчета
            session_data = {
                'processed_data': self.processed_data.to_dict('records')
            }
            
            # Конвертация дат в строки
            for item in session_data['processed_data']:
                if hasattr(item['date'], 'strftime'):
                    item['date'] = item['date'].strftime('%Y-%m-%d')
            
            # Генерация PDF
            pdf_buffer = self.report_generator.generate_sales_report(session_data)
            
            if pdf_buffer:
                # Диалог сохранения файла
                file_path = filedialog.asksaveasfilename(
                    title="Сохранить отчет по продажам",
                    defaultextension=".pdf",
                    filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
                )
                
                if file_path:
                    # Сохранение PDF файла
                    with open(file_path, 'wb') as f:
                        f.write(pdf_buffer.getvalue())
                    self.logger.info(f"Отчет сохранен: {file_path}")
                    messagebox.showinfo("Успех", f"Отчет сохранен: {file_path}")
            else:
                messagebox.showerror("Ошибка", "Не удалось сгенерировать отчет")
                
        except Exception as e:
            # Обработка ошибок генерации отчета
            error_msg = f"Ошибка при генерации отчета: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)

    def generate_forecast_report(self):
        """Генерация PDF отчета по результатам прогнозирования"""
        # Проверка наличия результатов прогнозирования
        if self.forecast_results is None:
            messagebox.showwarning("Предупреждение", "Сначала выполните прогнозирование!")
            return
        
        try:
            # Подготовка данных для отчета
            session_data = {
                'forecast_results': self.forecast_results,
                'model_accuracy': self.model_accuracy
            }
            
            # Генерация PDF
            pdf_buffer = self.report_generator.generate_forecast_report(session_data)
            
            if pdf_buffer:
                # Диалог сохранения файла
                file_path = filedialog.asksaveasfilename(
                    title="Сохранить отчет по прогнозам",
                    defaultextension=".pdf",
                    filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
                )
                
                if file_path:
                    # Сохранение PDF файла
                    with open(file_path, 'wb') as f:
                        f.write(pdf_buffer.getvalue())
                    self.logger.info(f"Отчет по прогнозам сохранен: {file_path}")
                    messagebox.showinfo("Успех", f"Отчет сохранен: {file_path}")
            else:
                messagebox.showerror("Ошибка", "Не удалось сгенерировать отчет")
                
        except Exception as e:
            # Обработка ошибок генерации отчета
            error_msg = f"Ошибка при генерации отчета по прогнозам: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)

    def generate_full_report(self):
        """Генерация полного PDF отчета со всеми данными"""
        # Проверка наличия исторических данных
        if self.processed_data is None or self.processed_data.empty:
            messagebox.showwarning("Предупреждение", "Нет данных для генерации отчета!")
            return
        
        # Проверка наличия результатов прогнозирования
        if self.forecast_results is None:
            messagebox.showwarning("Предупреждение", "Сначала выполните прогнозирование!")
            return
        
        try:
            # Подготовка всех данных для отчета
            session_data = {
                'processed_data': self.processed_data.to_dict('records'),
                'forecast_results': self.forecast_results or [],
                'model_accuracy': self.model_accuracy
            }
            
            # Конвертация дат в строки
            for item in session_data['processed_data']:
                if hasattr(item['date'], 'strftime'):
                    item['date'] = item['date'].strftime('%Y-%m-%d')
            
            # Генерация полного PDF отчета
            pdf_buffer = self.report_generator.generate_full_report(session_data)
            
            if pdf_buffer:
                # Диалог сохранения файла
                file_path = filedialog.asksaveasfilename(
                    title="Сохранить полный отчет",
                    defaultextension=".pdf",
                    filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
                )
                
                if file_path:
                    # Сохранение PDF файла
                    with open(file_path, 'wb') as f:
                        f.write(pdf_buffer.getvalue())
                    self.logger.info(f"Полный отчет сохранен: {file_path}")
                    messagebox.showinfo("Успех", f"Отчет сохранен: {file_path}")
            else:
                messagebox.showerror("Ошибка", "Не удалось сгенерировать отчет")
                
        except Exception as e:
            # Обработка ошибок генерации отчета
            error_msg = f"Ошибка при генерации полного отчета: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)

def main():
    """Основная функция запуска приложения"""
    try:
        # Создание главного окна и запуск приложения
        root = tk.Tk()
        app = SalesForecastApp(root)
        root.mainloop()
    except Exception as e:
        # Обработка критических ошибок приложения
        logging.error(f"Критическая ошибка приложения: {str(e)}")
        messagebox.showerror("Критическая ошибка", f"Приложение завершилось с ошибкой:\n{str(e)}")

if __name__ == "__main__":
    main()