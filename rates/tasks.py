from celery import shared_task
from .data_collection.api_client import fetch_exchange_rates
from .data_collection.data_loader import save_exchange_rates
from .models import ExchangeRateNormalized, Currency

# Валюты, которые должны сохраняться
ALLOWED_CURRENCIES = {"USD", "CNY", "HUF", "PLN", "CZK", "GBP"}


@shared_task
def collect_and_save_exchange_rates():
    try:
        # Получение данных
        date, rates = fetch_exchange_rates()

        # Сохранение данных в ненормализованную таблицу
        save_exchange_rates(date, rates)

        # Сохранение данных в нормализованную таблицу
        for currency_code, rate_value in rates.items():
            if currency_code in ALLOWED_CURRENCIES:
             currency, _ = Currency.objects.get_or_create(currency_code=currency_code)
             ExchangeRateNormalized.objects.update_or_create(
                date=date,
                currency=currency,
                defaults={'rate_value': rate_value}
            )

        return f"Данные за {date} успешно сохранены в обе таблицы!"
    except Exception as e:
        return f"Ошибка при выполнении задачи: {e}"
