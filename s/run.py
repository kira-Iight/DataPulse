#!/usr/bin/env python3
"""
Скрипт запуска DataPulse Analytics
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import main
    print("🚀 Запуск DataPulse Analytics...")
    main()
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Убедитесь, что установлены все зависимости:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Ошибка при запуске: {e}")
    sys.exit(1)
