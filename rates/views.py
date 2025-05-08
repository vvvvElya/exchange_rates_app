from django.shortcuts import render
from .models import Prediction
from .forecasting.predictions import save_predictions
from .models import ExchangeRate
from rates.models import ExchangeRateNormalized
from datetime import datetime, timedelta
import plotly.graph_objs as go
import base64
import io
import urllib
import json
import pandas as pd
import decimal
from decimal import Decimal  # Импортируем Decimal
from .models import Prediction
from .forecasting.prophet_model import predict_prophet
from .forecasting.nixtla_model import predict_timegpt


# Импорт моделей предсказаний
from .forecasting.lstm_model import predict_future as predict_lstm
from .forecasting.linear_regression_model import predict_linear_regression
from .forecasting.predictions import save_predictions
# Маппинг моделей для удобства расширения
MODEL_FUNCTIONS = {
    'lstm': predict_lstm,
    'linear_regression': predict_linear_regression,
    'prophet': predict_prophet,
    'timegpt': predict_timegpt,
}


def home(request):
    return render(request, 'home.html')  # Указываем правильный путь к шаблону

def graph_view(request):
    # Получаем параметры из GET-запроса
    start_date = request.GET.get('start_date', '2024-01-01')
    end_date = request.GET.get('end_date', '2024-12-31')
    currency = request.GET.get('currency', 'USD').upper()

    # Преобразуем строки в даты
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)

    # Запрашиваем данные из базы
    data = ExchangeRate.objects.filter(date__range=[start_date, end_date]).order_by('date')

    # Подготавливаем данные для графика
    dates = [item.date.strftime('%Y-%m-%d') for item in data]
    values = [float(getattr(item, currency)) for item in data if getattr(item, currency) is not None]

    # Вычисляем минимальное, максимальное и среднее значение
    if values:
        min_value = min(values)
        max_value = max(values)
        avg_value = round(sum(values) / len(values), 4)  # Округление до 4 знаков
        min_date = dates[values.index(min_value)]
        max_date = dates[values.index(max_value)]
    else:
        min_value = max_value = avg_value = min_date = max_date = "Nie sú dostupné údaje"

    # Лог для отладки
    print(f"Min: {min_value} ({min_date}), Max: {max_value} ({max_date}), Avg: {avg_value}")

    context = {
        "dates": json.dumps(dates),
        "values": json.dumps(values),
        "currency": currency,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "min_value": min_value,
        "min_date": min_date,
        "max_value": max_value,
        "max_date": max_date,
        "avg_value": avg_value,
    }

    return render(request, "graph.html", context)


"""
def predictions_view(request):
    currency = request.GET.get('currency', 'USD')  # Выбираем валюту из формы
    model = request.GET.get('model', 'lstm')  # Выбираем метод предсказания
    days = int(request.GET.get('days', 10))  # Количество дней предсказания

    print(f"DEBUG: Получена валюта - {currency}, модель - {model}, дней - {days}")  # Проверяем, что приходит

    # Генерируем предсказания
    save_predictions(currency, model, days)

    # Проверяем, что model используется везде правильно
    predictions = Prediction.objects.filter(
        currency_code__currency_code=currency,  # фильтруем по внешнему ключу
        model_name=model
    ).order_by('-date')[:days]

    return render(request, 'predictions.html', {
        'currency': currency,
        'model': model,
        'days': days,
        'predictions': predictions
    })
"""
def predictions_view(request):
    currency = request.GET.get('currency')
    model = request.GET.get('model')
    days = int(request.GET.get('days', 10))

    print(f"📢 Запрос предсказаний для {currency}, модель: {model}, дней: {days}")

    try:
        save_predictions(currency, model, days)

        predictions_qs = Prediction.objects.filter(
            currency_code__currency_code=currency,
            model_name=model
        ).order_by('date')

        if not predictions_qs.exists():
            raise ValueError(f'Нет предсказаний в базе для {currency} ({model})')

        historical_qs = ExchangeRateNormalized.objects.filter(currency__currency_code=currency).order_by('date')
        df_hist = pd.DataFrame(list(historical_qs.values()))

        if df_hist.empty:
            raise ValueError(f'Нет исторических данных для {currency}')

        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_hist = df_hist.sort_values("date")

        # ✅ Правильное разбиение данных по дате для совместимости с Prophet
        split_date = df_hist['date'].iloc[int(len(df_hist)*0.9) - 1]
        df_train = df_hist[df_hist['date'] <= split_date]
        df_test = df_hist[df_hist['date'] > split_date]

        model_function = MODEL_FUNCTIONS.get(model)
        if not model_function:
            raise ValueError(f"Модель {model} не поддерживается.")

        result = model_function(currency, days)

        if not result or 'test_result' not in result or 'future' not in result:
            raise ValueError(f'Модель "{model}" не вернула данные в ожидаемом формате')

        # ✅ Готовим данные для графика, учитывая доверительный интервал
        full_chart_data = {
            "train_dates": df_train["date"].dt.strftime('%Y-%m-%d').tolist(),
            "train_values": df_train["rate_value"].tolist(),
            "test_dates": result["test_result"]["dates"],
            "test_values": result["test_result"]["y_true"],
            "predicted_values": result["test_result"]["y_pred"],
            "future_dates": result["future"].index.strftime('%Y-%m-%d').tolist(),
            "future_values": result["future"].tolist(),
        }

        # ✅ Если модель возвращает доверительный интервал — добавляем его в график
        if "future_lower" in result and "future_upper" in result:
            full_chart_data["future_lower"] = result["future_lower"].tolist()
            full_chart_data["future_upper"] = result["future_upper"].tolist()

        def convert_decimal(obj):
            if isinstance(obj, decimal.Decimal):
                return float(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        return render(request, 'predictions.html', {
            'currency': currency,
            'model': model,
            'days': days,
            'predictions': predictions_qs,
            'full_chart_data': json.dumps(full_chart_data, default=convert_decimal),
            'metrics': result["test_result"]["metrics"],
        })

    except Exception as e:
        return render(request, 'predictions.html', {
            'currency': currency,
            'model': model,
            'days': days,
            'error': f'❌ {str(e)}'
        })

from rates.data_collection.data_loader import backfill_missing_data  # поправь путь
from django.shortcuts import redirect

def backfill_view(request):
    if request.method == 'POST':
        backfill_missing_data(request)  # Вот так — передаём request!
        return redirect('exchange_rate_graph')
    return redirect('exchange_rate_graph')

def contact_view(request):
    return render(request, 'contact.html')
