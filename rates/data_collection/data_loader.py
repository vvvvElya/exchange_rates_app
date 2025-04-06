import pandas as pd
from django.db import connection
from rates.models import ExchangeRate, ExchangeRateNormalized, Currency
from datetime import datetime, timedelta
from rates.models import ExchangeRateNormalized
import requests
import datetime
from django.contrib import messages



def save_exchange_rates(date, rates):
    last_known_rates = ExchangeRate.objects.order_by('-date').first()

    # Если записи за сегодня нет, создаём её
    if not ExchangeRate.objects.filter(date=date).exists():
        ExchangeRate.objects.create(
            date=date,
            USD=rates.get('USD', last_known_rates.USD if last_known_rates else None),
            CNY=rates.get('CNY', last_known_rates.CNY if last_known_rates else None),
            HUF=rates.get('HUF', last_known_rates.HUF if last_known_rates else None),
            PLN=rates.get('PLN', last_known_rates.PLN if last_known_rates else None),
            CZK=rates.get('CZK', last_known_rates.CZK if last_known_rates else None),
            GBP=rates.get('GBP', last_known_rates.GBP if last_known_rates else None),
        )
        print(f"✅ Данные за {date} успешно сохранены.")
    else:
        print(f"🔹 Данные за {date} уже есть в базе, обновление не требуется.")

    # **Заполняем пропущенные даты**
    fill_missing_dates()


def fill_missing_dates():

    all_dates = pd.date_range(start="2024-01-01", end=datetime.today())
    existing_dates = set(ExchangeRate.objects.values_list('date', flat=True))

    missing_dates = [d for d in all_dates if d.date() not in existing_dates]

    if missing_dates:
        last_known_rates = ExchangeRate.objects.order_by('-date').first()
        for missing_date in missing_dates:
            ExchangeRate.objects.create(
                date=missing_date,
                USD=last_known_rates.USD,
                CNY=last_known_rates.CNY,
                HUF=last_known_rates.HUF,
                PLN=last_known_rates.PLN,
                CZK=last_known_rates.CZK,
                GBP=last_known_rates.GBP
            )


def load_exchange_rates(currency_code):
    #queryset = ExchangeRateNormalized.objects.filter(currency_code=currency_code).order_by('date')
    queryset = ExchangeRateNormalized.objects.filter(currency__currency_code=currency_code).order_by('date')
   # queryset = ExchangeRateNormalized.objects.filter(currency_id=currency_code).order_by('date')
    df = pd.DataFrame.from_records(
        queryset.values('date', 'rate_value'),
        index='date'
    )
    return df



# новая функция

BASE_CURRENCY = 'EUR'
TARGET_CURRENCIES = ['USD', 'CNY', 'HUF', 'PLN', 'CZK', 'GBP']

def fetch_and_save_exchange_rates(date):
    # Просто проверяем, если данные есть — выходим молча
    if ExchangeRate.objects.filter(date=date).exists():
        return

    date_str = date.strftime('%Y-%m-%d')
    url = f"https://api.frankfurter.app/{date_str}?from={BASE_CURRENCY}&to={','.join(TARGET_CURRENCIES)}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json().get('rates', {})
        if data:
            ExchangeRate.objects.update_or_create(
                date=date,
                defaults={
                    'USD': data.get('USD'),
                    'CNY': data.get('CNY'),
                    'HUF': data.get('HUF'),
                    'PLN': data.get('PLN'),
                    'CZK': data.get('CZK'),
                    'GBP': data.get('GBP'),
                }
            )
        else:
            print(f"⚠️ Нет данных за {date_str}.")
    else:
        print(f"❌ Ошибка при запросе данных за {date_str}: {response.status_code}")

def backfill_missing_data(request=None):
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date.today()

    # Шаг 1: Сбор данных с API Frankfurter и сохранение, только если их нет в базе
    current_date = start_date
    while current_date <= end_date:
        fetch_and_save_exchange_rates(current_date)
        current_date += datetime.timedelta(days=1)

    # Шаг 2: Переносим данные в exchange_rates_normalized
    exchange_rates = ExchangeRate.objects.filter(date__range=[start_date, end_date]).order_by('date')

    for rate in exchange_rates:
        for currency in TARGET_CURRENCIES:
            try:
                currency_obj = Currency.objects.get(currency_code=currency)
                ExchangeRateNormalized.objects.update_or_create(
                    date=rate.date,
                    currency=currency_obj,
                    defaults={'rate_value': getattr(rate, currency)}
                )
            except Currency.DoesNotExist:
                print(f"⚠️ Валюта {currency} не найдена в таблице Currency!")

    # Шаг 3: Уведомление на сайт
    if request:
        messages.success(request, "✅ Dáta boli úspešne aktualizované!")

# Как использовать:
# backfill_missing_data()
