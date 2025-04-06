import os
import django

# Настройка Django окружения
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'exchange_rates.settings')
django.setup()

# Импорт функций для работы с данными
from rates.data_collection.data_loader import save_exchange_rates
from rates.data_collection.api_client import fetch_exchange_rates

if __name__ == "__main__":
    try:
        # Сбор данных с API Европейского центрального банка
        date, rates = fetch_exchange_rates()
        # Сохранение данных в базу
        save_exchange_rates(date, rates)
    except Exception as e:
        print(f"Ошибка: {e}")
