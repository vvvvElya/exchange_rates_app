from rates.forecasting.lstm_model import predict_future
from rates.forecasting.linear_regression_model import predict_linear_regression
from rates.forecasting.prophet_model import predict_prophet
from rates.forecasting.nixtla_model import predict_timegpt
import pandas as pd
from rates.models import Prediction, Currency
from datetime import datetime


def save_predictions(currency, model_name="prophet", days=10):
    """Сохраняет предсказания в базу данных, избегая дублирования."""

    print(f"⚡ Генерируем предсказания для {currency}, модель: {model_name}, {days} дней...")

    # Удаляем старые предсказания перед добавлением новых
    Prediction.objects.filter(currency_code__currency_code=currency, model_name=model_name).delete()

    # Выбираем метод предсказания
    if model_name == "lstm":
        result = predict_future(currency, days, "lstm")
        predictions = result["future"]  # это Series с индексом datetime и значениями
        dates = predictions.index
        values = predictions.values



    elif model_name == "linear_regression":
        result = predict_linear_regression(currency, days)
        predictions = result["future"]  # <-- Берем только прогноз на будущее
        dates = predictions.index
        values = predictions.values

    elif model_name == "prophet":
        result = predict_prophet(currency, days)
        predictions = result["future"]
        dates = predictions.index
        values = predictions.values

    elif model_name == "timegpt":
        result = predict_timegpt(currency, days)
        predictions = result["future"]
        dates = predictions.index
        values = predictions.values

    else:
        print(f" Неизвестный метод предсказания: {model_name}")
        return

    if predictions.empty:
        print(f"⚠ Нет предсказаний для {currency} ({model_name})")
        return

    currency_obj, _ = Currency.objects.get_or_create(currency_code=currency)

    # 🔍 Проверяем, есть ли предыдущие предсказания
    last_real_data = Prediction.objects.filter(currency_code=currency_obj).order_by("-date").first()
    last_real_date = last_real_data.date if last_real_data else datetime.today().date()

    print(f" Последняя дата с реальными данными: {last_real_date}")

    # Проходим по предсказаниям и сохраняем
    for date_value, value in zip(dates[:days], values[:days]):
        if isinstance(date_value, pd.Timestamp):
            date_value = date_value.date()
        elif isinstance(date_value, datetime):
            date_value = date_value.date()
        elif isinstance(date_value, str):
            date_value = datetime.strptime(date_value, "%Y-%m-%d").date()

        print(f"Сохраняем предсказание: {date_value} - {value} ({model_name})")

        Prediction.objects.create(
            date=date_value,
            currency_code=currency_obj,
            predicted_value=float(value),
            model_name=model_name,
        )

    print(f"Успешно сохранены предсказания для {currency} ({days} дней) по модели {model_name}")


if __name__ == "__main__":
    save_predictions("USD", "linear_regression")  # Тестируем линейную регрессию
