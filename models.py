# models/models.py
from sqlalchemy import MetaData, Table, Column, Integer, String, Date, Numeric, Boolean, TIMESTAMP, Float, DateTime
from sqlalchemy.sql import func
from datetime import datetime

metadata = MetaData()

# ОПИСАНИЕ ТАБЛИЦ
raw_data_table = Table('raw_sales_data', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('date', Date, nullable=False),
    Column('product_name', String(255)),
    Column('quantity', Integer),
    Column('price', Numeric(10, 2)),
    Column('created_at', TIMESTAMP, server_default=func.now())
)

processed_data_table = Table('processed_data', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('date', Date, nullable=False, unique=True),
    Column('total_sales', Numeric(15, 2)),  # Исправлено: было total_daily_sales, должно быть total_sales
    Column('day_of_week', Integer),
    Column('month', Integer),
    Column('is_holiday', Boolean, default=False),
    Column('created_at', TIMESTAMP, server_default=func.now())
)

forecast_results_table = Table('forecast_results', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('forecast_date', Date, nullable=False),
    Column('target_date', Date, nullable=False),
    Column('predicted_amount', Numeric(15, 2)),
    Column('model_name', String(100)),
    Column('confidence_interval', String(100)),
    Column('created_at', TIMESTAMP, server_default=func.now())
)

model_accuracy_table = Table(
    'model_accuracy',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('accuracy', Float),
    Column('created_at', DateTime, default=datetime.now),
    Column('model_name', String(50))
)