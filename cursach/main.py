# main.py (обновленная версия)
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

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем новые модули
from config import AppConfig
from data_manager import DataManager
from ml_engine import ForecastEngine
from logging_config import setup_logging
from report_generator import generate_sales_report, generate_forecast_report, generate_full_report

# Инициализируем логирование
setup_logging()

class ModernTheme:
    """Современная цветовая схема и стили"""
    COLORS = AppConfig.COLORS
    FONTS = AppConfig.FONTS

class SalesForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DataPulse Analytics - Система прогнозирования продаж")
        self.root.geometry("1400x900")
        self.root.configure(bg=ModernTheme.COLORS['background'])
        
        # Инициализируем менеджеры
        self.data_manager = DataManager()
        self.forecast_engine = ForecastEngine()
        self.logger = logging.getLogger(__name__)
        
        # Устанавливаем иконку приложения (если есть)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
        
        # Настраиваем стили
        self.setup_styles()
        
        # Данные приложения
        self.raw_data = None
        self.processed_data = None
        self.forecast_results = None
        self.model_accuracy = []
        
        # Создаем интерфейс
        self.create_widgets()
        
        self.logger.info("Приложение инициализировано")
        
    def setup_styles(self):
        """Настраивает современные стили для виджетов"""
        style = ttk.Style()
        
        # Современная тема
        style.theme_use('clam')
        
        # Настраиваем цвета
        style.configure('TFrame', background=ModernTheme.COLORS['background'])
        style.configure('TLabel', background=ModernTheme.COLORS['background'], font=ModernTheme.FONTS['normal'])
        style.configure('TButton', font=ModernTheme.FONTS['normal'], padding=6)
        style.configure('Primary.TButton', background=ModernTheme.COLORS['primary'], foreground='white')
        style.configure('Secondary.TButton', background=ModernTheme.COLORS['secondary'], foreground='white')
        style.configure('Success.TButton', background=ModernTheme.COLORS['success'], foreground='white')
        
        # Стиль для карточек
        style.configure('Card.TFrame', background=ModernTheme.COLORS['card'], relief='raised', borderwidth=1)
        
        # Стиль для Treeview
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
        
        style.map('Treeview.Heading', 
                 background=[('active', ModernTheme.COLORS['primary_light'])])
        
        # Настройка Notebook
        style.configure('TNotebook', background=ModernTheme.COLORS['background'], borderwidth=0)
        style.configure('TNotebook.Tab', 
                       background=ModernTheme.COLORS['secondary'],
                       foreground='white',
                       padding=[15, 5],
                       font=ModernTheme.FONTS['normal'])
        
        style.map('TNotebook.Tab', 
                 background=[('selected', ModernTheme.COLORS['primary']),
                           ('active', ModernTheme.COLORS['primary_light'])])
        
    def create_widgets(self):
        """Создает современный интерфейс"""
        # Главный контейнер
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Заголовок приложения
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, 
                 text="📊 DataPulse Analytics", 
                 font=ModernTheme.FONTS['title'],
                 foreground=ModernTheme.COLORS['primary']).pack(side=tk.LEFT)
        
        ttk.Label(header_frame, 
                 text="Система прогнозирования продаж", 
                 font=ModernTheme.FONTS['subtitle'],
                 foreground=ModernTheme.COLORS['secondary']).pack(side=tk.LEFT, padx=(10, 0))
        
        # Основной контент
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Боковая панель
        sidebar_frame = ttk.Frame(content_frame, width=250, style='Card.TFrame')
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        sidebar_frame.pack_propagate(False)
        
        # Основная область
        main_area_frame = ttk.Frame(content_frame)
        main_area_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Заполняем боковую панель
        self.create_sidebar(sidebar_frame)
        
        # Заполняем основную область
        self.create_main_area(main_area_frame)
        
    def create_sidebar(self, parent):
        """Создает боковую панель с кнопками"""
        # Заголовок боковой панели
        ttk.Label(parent, 
                 text="Панель управления", 
                 font=ModernTheme.FONTS['subtitle'],
                 background=ModernTheme.COLORS['card']).pack(pady=20)
        
        # Кнопки управления данными
        ttk.Label(parent, 
                 text="Данные", 
                 font=ModernTheme.FONTS['normal'],
                 background=ModernTheme.COLORS['card'],
                 foreground=ModernTheme.COLORS['secondary']).pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        ttk.Button(parent, 
                  text="📁 Загрузить CSV", 
                  command=self.load_csv_file,
                  style='Primary.TButton').pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Button(parent, 
                  text="🗑️ Очистить данные", 
                  command=self.clear_data,
                  style='Secondary.TButton').pack(fill=tk.X, padx=20, pady=5)
        
        # Кнопки прогнозирования
        ttk.Label(parent, 
                 text="Прогнозирование", 
                 font=ModernTheme.FONTS['normal'],
                 background=ModernTheme.COLORS['card'],
                 foreground=ModernTheme.COLORS['secondary']).pack(anchor=tk.W, padx=20, pady=(20, 5))
        
        ttk.Button(parent, 
                  text="🧮 Рассчитать прогноз", 
                  command=self.run_forecast,
                  style='Success.TButton').pack(fill=tk.X, padx=20, pady=5)
        
        # Кнопки отчетов
        ttk.Label(parent, 
                 text="Отчеты", 
                 font=ModernTheme.FONTS['normal'],
                 background=ModernTheme.COLORS['card'],
                 foreground=ModernTheme.COLORS['secondary']).pack(anchor=tk.W, padx=20, pady=(20, 5))
        
        ttk.Button(parent, 
                  text="📈 Отчет по продажам", 
                  command=self.generate_sales_report).pack(fill=tk.X, padx=20, pady=2)
        
        ttk.Button(parent, 
                  text="🔮 Отчет по прогнозам", 
                  command=self.generate_forecast_report).pack(fill=tk.X, padx=20, pady=2)
        
        ttk.Button(parent, 
                  text="📊 Полный отчет", 
                  command=self.generate_full_report).pack(fill=tk.X, padx=20, pady=2)
        
        # Статус бар внизу
        status_frame = ttk.Frame(parent, style='Card.TFrame')
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.status_label = ttk.Label(status_frame, 
                                     text="Готов к работе", 
                                     font=ModernTheme.FONTS['small'],
                                     background=ModernTheme.COLORS['card'],
                                     foreground=ModernTheme.COLORS['secondary'])
        self.status_label.pack(pady=5)
        
    def create_main_area(self, parent):
        """Создает основную область с вкладками"""
        # Создаем Notebook с вкладками
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка "Данные"
        self.data_frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.data_frame, text="📊 Данные")
        
        # Вкладка "Прогноз"
        self.forecast_frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.forecast_frame, text="🔮 Прогноз")
        
        # Вкладка "Статистика"
        self.stats_frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.stats_frame, text="📈 Статистика")
        
        # Вкладка "Информация"
        self.info_frame = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(self.info_frame, text="ℹ️ Информация")
        
        # Заполняем вкладки
        self.create_data_tab()
        self.create_forecast_tab()
        self.create_stats_tab()
        self.create_info_tab()
        
    def create_data_tab(self):
        """Создает вкладку с данными"""
        # Заголовок
        header = ttk.Frame(self.data_frame)
        header.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header, 
                 text="Исторические данные продаж", 
                 font=ModernTheme.FONTS['subtitle']).pack(side=tk.LEFT)
        
        # Информация о данных
        self.data_info_label = ttk.Label(header, 
                                        text="Данные не загружены", 
                                        font=ModernTheme.FONTS['small'],
                                        foreground=ModernTheme.COLORS['secondary'])
        self.data_info_label.pack(side=tk.RIGHT)
        
        # Таблица данных в карточке
        card_frame = ttk.Frame(self.data_frame, style='Card.TFrame')
        card_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем Treeview для таблицы
        columns = ('Дата', 'Продажи', 'День недели', 'Месяц', 'Квартал', 'Выходной', 'Праздник')
        self.data_tree = ttk.Treeview(card_frame, columns=columns, show='headings', height=20)
        
        # Настройка колонок
        column_widths = {'Дата': 100, 'Продажи': 120, 'День недели': 90, 
                        'Месяц': 70, 'Квартал': 70, 'Выходной': 70, 'Праздник': 70}
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=column_widths.get(col, 100), anchor=tk.CENTER)
        
        # Скроллбар
        scrollbar = ttk.Scrollbar(card_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # Размещение
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
    def create_forecast_tab(self):
        """Создает вкладку с прогнозом"""
        # Заголовок
        header = ttk.Frame(self.forecast_frame)
        header.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header, 
                 text="Прогноз продаж на 7 дней", 
                 font=ModernTheme.FONTS['subtitle']).pack(side=tk.LEFT)
        
        # Информация о прогнозе
        self.forecast_info_label = ttk.Label(header, 
                                           text="Прогноз не выполнен", 
                                           font=ModernTheme.FONTS['small'],
                                           foreground=ModernTheme.COLORS['secondary'])
        self.forecast_info_label.pack(side=tk.RIGHT)
        
        # График прогноза в карточке
        card_frame = ttk.Frame(self.forecast_frame, style='Card.TFrame')
        card_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем график
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.ax.set_title("Прогноз продаж", fontsize=14, fontweight='bold', pad=20)
        self.ax.set_xlabel("Дата", fontsize=12)
        self.ax.set_ylabel("Продажи (руб.)", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # Настройка стиля графика
        self.fig.patch.set_facecolor(ModernTheme.COLORS['card'])
        self.ax.set_facecolor(ModernTheme.COLORS['card'])
        
        # Canvas для matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, card_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_stats_tab(self):
        """Создает вкладку со статистикой"""
        # Заголовок
        ttk.Label(self.stats_frame, 
                 text="Статистика данных", 
                 font=ModernTheme.FONTS['subtitle']).pack(anchor=tk.W, pady=(0, 15))
        
        # Карточки с метриками
        metrics_frame = ttk.Frame(self.stats_frame)
        metrics_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.metrics = {}
        metrics_data = [
            ("Всего записей", "total_records", "0", ModernTheme.COLORS['primary']),
            ("Общий объем продаж", "total_sales", "0 руб.", ModernTheme.COLORS['success']),
            ("Среднедневной объем", "avg_daily", "0 руб.", ModernTheme.COLORS['warning']),
            ("Максимальные продажи", "max_sales", "0 руб.", ModernTheme.COLORS['danger']),
            ("Минимальные продажи", "min_sales", "0 руб.", ModernTheme.COLORS['secondary']),
            ("Точность модели", "model_accuracy", "N/A", ModernTheme.COLORS['primary']),
            ("Период данных", "date_range", "N/A", ModernTheme.COLORS['success'])
        ]
        
        for i, (label, key, default, color) in enumerate(metrics_data):
            row = i // 3
            col = i % 3
            metric_card = ttk.Frame(metrics_frame, style='Card.TFrame', width=200, height=100)
            metric_card.grid(row=row, column=col, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
            metric_card.grid_propagate(False)
            
            ttk.Label(metric_card, 
                     text=label, 
                     font=ModernTheme.FONTS['small'],
                     background=ModernTheme.COLORS['card'],
                     foreground=ModernTheme.COLORS['secondary']).pack(pady=(15, 5))
            
            value_label = ttk.Label(metric_card, 
                                  text=default, 
                                  font=ModernTheme.FONTS['metric'],
                                  background=ModernTheme.COLORS['card'],
                                  foreground=color)
            value_label.pack(pady=(0, 15))
            
            self.metrics[key] = value_label
        
        # График статистики
        card_frame = ttk.Frame(self.stats_frame, style='Card.TFrame')
        card_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stats_fig, self.stats_ax = plt.subplots(figsize=(12, 6))
        self.stats_ax.set_title("Динамика продаж", fontsize=14, fontweight='bold', pad=20)
        self.stats_ax.set_xlabel("Дата", fontsize=12)
        self.stats_ax.set_ylabel("Продажи (руб.)", fontsize=12)
        self.stats_ax.grid(True, alpha=0.3)
        
        # Настройка стиля графика
        self.stats_fig.patch.set_facecolor(ModernTheme.COLORS['card'])
        self.stats_ax.set_facecolor(ModernTheme.COLORS['card'])
        
        # Canvas для статистики
        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, card_frame)
        self.stats_canvas.draw()
        self.stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_info_tab(self):
        """Создает информационную вкладку"""
        # Карточка с информацией
        card_frame = ttk.Frame(self.info_frame, style='Card.TFrame')
        card_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=50)
        
        info_text = """
        🚀 DataPulse Analytics - Система прогнозирования продаж v2.0
        
        📋 Улучшенные возможности:
        • Расширенная валидация и обработка данных
        • Кэширование моделей для ускорения работы
        • Расширенные признаки для улучшения прогнозов
        • Детальное логирование и обработка ошибок
        • Бизнес-метрики и аналитика
        
        🔧 Технические улучшения:
        • Модульная архитектура
        • Валидация данных с помощью Pydantic
        • Кэширование с LRU и joblib
        • Расширенное логирование с ротацией
        • Оптимизированные алгоритмы ML
        
        📊 Поддерживаемые форматы данных:
        • CSV файлы с колонками: date, quantity, price
        • Дата в формате YYYY-MM-DD
        • Числовые значения для quantity и price
        
        🎯 Алгоритм работы:
        1. Загрузите CSV файл с данными продаж
        2. Система автоматически обработает и проверит данные
        3. Нажмите "Рассчитать прогноз" для ML анализа
        4. Просматривайте результаты на вкладках
        5. Генерируйте отчеты в нужном формате
        
        ⚙️ Технологии:
        • Python 3.8+
        • Scikit-learn для машинного обучения
        • Pandas для обработки данных
        • Matplotlib для визуализации
        • WeasyPrint для генерации PDF
        
        📞 Поддержка:
        Для вопросов и предложений обращайтесь к разработчикам системы.
        """
        
        text_widget = tk.Text(card_frame, 
                             wrap=tk.WORD, 
                             font=ModernTheme.FONTS['normal'],
                             background=ModernTheme.COLORS['card'],
                             foreground=ModernTheme.COLORS['dark'],
                             borderwidth=0,
                             padx=20,
                             pady=20)
        
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
    def update_status(self, message, color=ModernTheme.COLORS['secondary']):
        """Обновляет статус бар"""
        self.status_label.config(text=message, foreground=color)
        self.logger.info(f"Статус: {message}")
        
    def load_csv_file(self):
        """Загружает CSV файл с данными продаж"""
        self.update_status("Выбор файла CSV...")
        
        file_path = filedialog.askopenfilename(
            title="Выберите CSV файл",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.update_status("Загрузка данных...", ModernTheme.COLORS['primary'])
                
                # Используем DataManager для загрузки и обработки
                self.raw_data = self.data_manager.load_data_from_csv(file_path)
                self.processed_data = self.data_manager.preprocess_data(self.raw_data)
                
                # Обновляем интерфейс
                self.update_data_table()
                self.update_stats()
                self.update_stats_chart()
                
                success_msg = f"Файл загружен успешно! Обработано {len(self.processed_data)} записей"
                self.update_status(success_msg, ModernTheme.COLORS['success'])
                messagebox.showinfo("Успех", success_msg)
                
            except Exception as e:
                error_msg = f"Ошибка при загрузке файла: {str(e)}"
                self.update_status("Ошибка при загрузке файла", ModernTheme.COLORS['danger'])
                self.logger.error(error_msg)
                messagebox.showerror("Ошибка", error_msg)
        else:
            self.update_status("Отменено", ModernTheme.COLORS['secondary'])
    
    def update_data_table(self):
        """Обновляет таблицу с данными"""
        # Очищаем таблицу
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if self.processed_data is not None and not self.processed_data.empty:
            # Получаем статистику
            stats = self.data_manager.get_data_statistics(self.processed_data)
            
            # Обновляем информацию о данных
            info_text = f"Записей: {stats['total_records']} | Период: {stats['date_range']['start']} - {stats['date_range']['end']}"
            self.data_info_label.config(text=info_text)
            
            # Заполняем таблицу
            for _, row in self.processed_data.iterrows():
                day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
                day_name = day_names[int(row['day_of_week'])] if 'day_of_week' in row else 'Н/Д'
                is_weekend = 'Да' if row.get('is_weekend', False) else 'Нет'
                is_holiday = 'Да' if row.get('is_holiday', False) else 'Нет'
                
                self.data_tree.insert('', 'end', values=(
                    str(row['date'].date()),
                    f"{row['total_sales']:,.2f}",
                    day_name,
                    int(row['month']),
                    f"Q{int(row['quarter'])}",
                    is_weekend,
                    is_holiday
                ))
        else:
            self.data_info_label.config(text="Данные не загружены")
    
    def update_stats(self):
        """Обновляет статистику"""
        if self.processed_data is not None and not self.processed_data.empty:
            stats = self.data_manager.get_data_statistics(self.processed_data)
            
            self.metrics['total_records'].config(text=str(stats['total_records']))
            self.metrics['total_sales'].config(text=f"{stats['total_sales']:,.2f} руб.")
            self.metrics['avg_daily'].config(text=f"{stats['avg_daily']:,.2f} руб.")
            self.metrics['max_sales'].config(text=f"{stats['max_sales']:,.2f} руб.")
            self.metrics['min_sales'].config(text=f"{stats['min_sales']:,.2f} руб.")
            self.metrics['date_range'].config(text=f"{stats['date_range']['days']} дней")
            
            # Обновляем точность модели если есть
            if self.model_accuracy:
                self.update_accuracy_metric()
    
    def update_stats_chart(self):
        """Обновляет график статистики"""
        if self.processed_data is not None and not self.processed_data.empty:
            self.stats_ax.clear()
            
            dates = self.processed_data['date']
            sales = self.processed_data['total_sales']
            
            # Используем современные цвета для графика
            self.stats_ax.plot(dates, sales, marker='o', linewidth=2.5, 
                             color=ModernTheme.COLORS['primary'], markersize=4, alpha=0.8)
            
            # Добавляем скользящее среднее
            if len(sales) >= 7:
                rolling_mean = sales.rolling(window=7).mean()
                self.stats_ax.plot(dates, rolling_mean, linewidth=2, 
                                 color=ModernTheme.COLORS['warning'], linestyle='--', alpha=0.7)
            
            self.stats_ax.set_title("Динамика продаж", fontsize=14, fontweight='bold', pad=20)
            self.stats_ax.set_xlabel("Дата", fontsize=12)
            self.stats_ax.set_ylabel("Продажи (руб.)", fontsize=12)
            self.stats_ax.grid(True, alpha=0.3)
            self.stats_ax.legend(['Фактические данные', 'Скользящее среднее (7 дней)'])
            
            # Поворачиваем подписи дат
            plt.setp(self.stats_ax.xaxis.get_majorticklabels(), rotation=45)
            
            self.stats_canvas.draw()
    
    def run_forecast(self):
        """Запускает прогнозирование"""
        if self.processed_data is None or self.processed_data.empty:
            self.update_status("Сначала загрузите данные!", ModernTheme.COLORS['warning'])
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        
        if len(self.processed_data) < AppConfig.MIN_DATA_POINTS:
            self.update_status("Недостаточно данных для прогноза", ModernTheme.COLORS['warning'])
            messagebox.showwarning("Предупреждение", 
                                 f"Недостаточно данных для прогнозирования!\nНужно минимум {AppConfig.MIN_DATA_POINTS} записей.")
            return
        
        # Показываем прогресс
        progress = tk.Toplevel(self.root)
        progress.title("Обучение модели")
        progress.geometry("400x150")
        progress.configure(bg=ModernTheme.COLORS['background'])
        progress.transient(self.root)
        progress.grab_set()
        
        # Центрируем окно прогресса
        progress.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - progress.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - progress.winfo_height()) // 2
        progress.geometry(f"+{x}+{y}")
        
        ttk.Label(progress, 
                 text="Обучение модели машинного обучения...", 
                 font=ModernTheme.FONTS['normal'],
                 background=ModernTheme.COLORS['background']).pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress, mode='indeterminate', length=300)
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_bar.start()
        
        self.update_status("Обучение модели...", ModernTheme.COLORS['primary'])
        
        def train_and_predict():
            try:
                # Подготавливаем данные для сессии
                session_data = {
                    'processed_data': self.processed_data.to_dict('records'),
                    'model_accuracy': self.model_accuracy
                }
                
                # Конвертируем даты в строки для сериализации
                for item in session_data['processed_data']:
                    if hasattr(item['date'], 'strftime'):
                        item['date'] = item['date'].strftime('%Y-%m-%d')
                
                # Используем ForecastEngine для обучения
                model, accuracy = self.forecast_engine.train_model(session_data)
                
                if model is not None:
                    # Делаем прогноз
                    predictions = self.forecast_engine.make_predictions(model, session_data, days_to_forecast=7)
                    
                    if predictions:
                        self.forecast_results = predictions
                        self.model_accuracy = session_data.get('model_accuracy', [])
                        
                        # Обновляем интерфейс в главном потоке
                        self.root.after(0, self.update_forecast_chart)
                        self.root.after(0, self.update_accuracy_metric)
                        self.root.after(0, progress.destroy)
                        
                        accuracy_percent = (1 - accuracy) * 100
                        success_msg = f"Прогноз выполнен! Точность: {accuracy_percent:.1f}%"
                        self.update_status(success_msg, ModernTheme.COLORS['success'])
                        self.root.after(0, lambda: messagebox.showinfo("Успех", success_msg))
                    else:
                        self.root.after(0, progress.destroy)
                        self.update_status("Ошибка создания прогноза", ModernTheme.COLORS['danger'])
                        self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не удалось создать прогноз"))
                else:
                    self.root.after(0, progress.destroy)
                    self.update_status("Ошибка обучения модели", ModernTheme.COLORS['danger'])
                    self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не удалось обучить модель"))
                    
            except Exception as e:
                self.root.after(0, progress.destroy)
                error_msg = f"Ошибка при прогнозировании: {str(e)}"
                self.update_status("Ошибка при прогнозировании", ModernTheme.COLORS['danger'])
                self.logger.error(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))
        
        # Запускаем в отдельном потоке
        thread = threading.Thread(target=train_and_predict)
        thread.daemon = True
        thread.start()
    
    def update_forecast_chart(self):
        """Обновляет график прогноза"""
        if self.forecast_results:
            self.ax.clear()
            
            # Обновляем информацию о прогнозе
            total_forecast = sum(pred['predicted_sales'] for pred in self.forecast_results)
            self.forecast_info_label.config(text=f"Прогноз: {total_forecast:,.0f} руб.")
            
            # Исторические данные
            if self.processed_data is not None and not self.processed_data.empty:
                dates = self.processed_data['date']
                sales = self.processed_data['total_sales']
                self.ax.plot(dates, sales, 
                           label='Исторические данные', 
                           marker='o', 
                           linewidth=2.5, 
                           color=ModernTheme.COLORS['primary'],
                           markersize=4,
                           alpha=0.8)
            
            # Прогноз
            forecast_dates = [pd.to_datetime(pred['date']) for pred in self.forecast_results]
            forecast_sales = [pred['predicted_sales'] for pred in self.forecast_results]
            
            self.ax.plot(forecast_dates, forecast_sales, 
                       label='Прогноз', 
                       marker='s', 
                       linewidth=3, 
                       color=ModernTheme.COLORS['success'],
                       markersize=6)
            
            # Доверительный интервал
            upper_bound = [sales * 1.2 for sales in forecast_sales]
            lower_bound = [sales * 0.8 for sales in forecast_sales]
            self.ax.fill_between(forecast_dates, lower_bound, upper_bound, 
                               alpha=0.2, 
                               color=ModernTheme.COLORS['success'],
                               label='Доверительный интервал (±20%)')
            
            self.ax.set_title("Прогноз продаж на 7 дней", fontsize=14, fontweight='bold', pad=20)
            self.ax.set_xlabel("Дата", fontsize=12)
            self.ax.set_ylabel("Продажи (руб.)", fontsize=12)
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            # Поворачиваем подписи дат
            plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
            
            self.canvas.draw()
    
    def update_accuracy_metric(self):
        """Обновляет метрику точности"""
        if self.model_accuracy:
            metrics = self.forecast_engine.get_model_metrics({'model_accuracy': self.model_accuracy})
            if metrics:
                self.metrics['model_accuracy'].config(text=f"{metrics['accuracy_percent']:.1f}%")
    
    def clear_data(self):
        """Очищает все данные"""
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите очистить все данные?"):
            self.raw_data = None
            self.processed_data = None
            self.forecast_results = None
            self.model_accuracy = []
            
            # Очищаем интерфейс
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            self.ax.clear()
            self.ax.set_title("Прогноз продаж", fontsize=14, fontweight='bold', pad=20)
            self.ax.set_xlabel("Дата", fontsize=12)
            self.ax.set_ylabel("Продажи (руб.)", fontsize=12)
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
            
            self.stats_ax.clear()
            self.stats_ax.set_title("Динамика продаж", fontsize=14, fontweight='bold', pad=20)
            self.stats_ax.set_xlabel("Дата", fontsize=12)
            self.stats_ax.set_ylabel("Продажи (руб.)", fontsize=12)
            self.stats_ax.grid(True, alpha=0.3)
            self.stats_canvas.draw()
            
            # Сбрасываем метрики
            self.metrics['total_records'].config(text="0")
            self.metrics['total_sales'].config(text="0 руб.")
            self.metrics['avg_daily'].config(text="0 руб.")
            self.metrics['max_sales'].config(text="0 руб.")
            self.metrics['min_sales'].config(text="0 руб.")
            self.metrics['model_accuracy'].config(text="N/A")
            self.metrics['date_range'].config(text="N/A")
            
            # Сбрасываем информационные лейблы
            self.data_info_label.config(text="Данные не загружены")
            self.forecast_info_label.config(text="Прогноз не выполнен")
            
            self.update_status("Данные очищены", ModernTheme.COLORS['success'])
            messagebox.showinfo("Успех", "Данные очищены")
    
    def generate_sales_report(self):
        """Генерирует отчет по продажам"""
        if self.processed_data is None or self.processed_data.empty:
            messagebox.showwarning("Предупреждение", "Нет данных для генерации отчета!")
            return
        
        try:
            session_data = {
                'processed_data': self.processed_data.to_dict('records')
            }
            
            # Конвертируем даты в строки
            for item in session_data['processed_data']:
                if hasattr(item['date'], 'strftime'):
                    item['date'] = item['date'].strftime('%Y-%m-%d')
            
            pdf_buffer = generate_sales_report(session_data)
            
            if pdf_buffer:
                # Сохраняем файл
                file_path = filedialog.asksaveasfilename(
                    title="Сохранить отчет по продажам",
                    defaultextension=".pdf",
                    filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
                )
                
                if file_path:
                    with open(file_path, 'wb') as f:
                        f.write(pdf_buffer.getvalue())
                    self.logger.info(f"Отчет сохранен: {file_path}")
                    messagebox.showinfo("Успех", f"Отчет сохранен: {file_path}")
            else:
                messagebox.showerror("Ошибка", "Не удалось сгенерировать отчет")
                
        except Exception as e:
            error_msg = f"Ошибка при генерации отчета: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)
    
    def generate_forecast_report(self):
        """Генерирует отчет по прогнозам"""
        if self.forecast_results is None:
            messagebox.showwarning("Предупреждение", "Сначала выполните прогнозирование!")
            return
        
        try:
            session_data = {
                'forecast_results': self.forecast_results,
                'model_accuracy': self.model_accuracy
            }
            
            pdf_buffer = generate_forecast_report(session_data)
            
            if pdf_buffer:
                file_path = filedialog.asksaveasfilename(
                    title="Сохранить отчет по прогнозам",
                    defaultextension=".pdf",
                    filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
                )
                
                if file_path:
                    with open(file_path, 'wb') as f:
                        f.write(pdf_buffer.getvalue())
                    self.logger.info(f"Отчет по прогнозам сохранен: {file_path}")
                    messagebox.showinfo("Успех", f"Отчет сохранен: {file_path}")
            else:
                messagebox.showerror("Ошибка", "Не удалось сгенерировать отчет")
                
        except Exception as e:
            error_msg = f"Ошибка при генерации отчета по прогнозам: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)
    
    def generate_full_report(self):
        """Генерирует полный отчет"""
        if self.processed_data is None or self.processed_data.empty:
            messagebox.showwarning("Предупреждение", "Нет данных для генерации отчета!")
            return
        
        try:
            session_data = {
                'processed_data': self.processed_data.to_dict('records'),
                'forecast_results': self.forecast_results or [],
                'model_accuracy': self.model_accuracy
            }
            
            # Конвертируем даты в строки
            for item in session_data['processed_data']:
                if hasattr(item['date'], 'strftime'):
                    item['date'] = item['date'].strftime('%Y-%m-%d')
            
            pdf_buffer = generate_full_report(session_data)
            
            if pdf_buffer:
                file_path = filedialog.asksaveasfilename(
                    title="Сохранить полный отчет",
                    defaultextension=".pdf",
                    filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
                )
                
                if file_path:
                    with open(file_path, 'wb') as f:
                        f.write(pdf_buffer.getvalue())
                    self.logger.info(f"Полный отчет сохранен: {file_path}")
                    messagebox.showinfo("Успех", f"Отчет сохранен: {file_path}")
            else:
                messagebox.showerror("Ошибка", "Не удалось сгенерировать отчет")
                
        except Exception as e:
            error_msg = f"Ошибка при генерации полного отчета: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)

def main():
    """Основная функция приложения"""
    try:
        root = tk.Tk()
        app = SalesForecastApp(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Критическая ошибка приложения: {str(e)}")
        messagebox.showerror("Критическая ошибка", f"Приложение завершилось с ошибкой:\n{str(e)}")

if __name__ == "__main__":
    main()