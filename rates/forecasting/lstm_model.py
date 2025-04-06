import os
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "exchange_rates.settings")
import django

django.setup()

from rates.data_collection.data_loader import load_exchange_rates
from rates.models import ExchangeRateNormalized

# =====================
# Пути для моделей и скейлеров
# =====================

def get_model_dir():
    return os.path.join(django.conf.settings.BASE_DIR, 'rates', 'forecasting', 'trained_models')

# =====================
# Кэш загруженных моделей и скейлеров
# =====================

loaded_models = {}
loaded_scalers = {}

# =====================
# Работа со скейлером
# =====================

def load_scaler(currency):
    if currency not in loaded_scalers:
        print(f"🔄 Načítavam scaler pre {currency} (lazy-loading)...")
        scaler_path = os.path.join(get_model_dir(), f'scaler_{currency}.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Скейлер не найден для {currency} по пути {scaler_path}")
        with open(scaler_path, 'rb') as f:
            loaded_scalers[currency] = pickle.load(f)
    return loaded_scalers[currency]

# =====================
# Загрузка TFLite модели
# =====================

def load_tflite_model(currency):
    if currency not in loaded_models:
        print(f"🔄 Načítavam TFLite model pre {currency} (lazy-loading)...")
        model_path = os.path.join(get_model_dir(), f'lstm_{currency}.tflite')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite модель не найдена по пути {model_path}")
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        loaded_models[currency] = interpreter
    return loaded_models[currency]

# =====================
# Предсказание через TFLite
# =====================

def predict_with_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.flatten()[0]

# =====================
# Основная функция прогноза с TFLite
# =====================

def predict_future(currency, days, model_name="lstm", look_back=30):
    days = int(days)  # сразу приводим в int
    if days <= 0:
        raise ValueError("Количество дней прогноза должно быть положительным числом.")

    historical_data = ExchangeRateNormalized.objects.filter(
        currency__currency_code=currency
    ).order_by('date')

    df = pd.DataFrame(list(historical_data.values()))
    if df.empty:
        raise ValueError("Нет данных для предсказания.")

    df = df.sort_values(by="date")
    df["rate_value"] = df["rate_value"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    values = df["rate_value"].values.reshape(-1, 1)
    scaler = load_scaler(currency)
   # scaled_data = scaler.transform(values)
    scaled_data = scaler.transform(pd.DataFrame(values, columns=['rate_value']))

    # Создание окон
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("Недостаточно данных для формирования обучающей выборки.")

    train_size = int(len(X) * 0.8)
    X_test, y_test = X[train_size:], y[train_size:]
    X_test = X_test.reshape((-1, look_back, 1))

    interpreter = load_tflite_model(currency)

    # Предсказание на тестовой выборке
    y_pred_scaled = [predict_with_tflite(interpreter, sample.reshape(1, look_back, 1)) for sample in X_test]
    y_pred_scaled = np.array(y_pred_scaled)

    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    if len(df.index) > look_back + train_size:
        test_start = df.index[look_back + train_size]
    else:
        test_start = df.index[-len(y_test):][0]

    test_dates = pd.date_range(start=test_start, periods=len(y_test))

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Прогноз на будущее
    last_window = scaled_data[-look_back:].reshape(1, look_back, 1)
    future_preds_scaled = []
    for _ in range(days):
        pred_scaled = predict_with_tflite(interpreter, last_window)
        future_preds_scaled.append(pred_scaled)
        last_window = np.append(last_window[:, 1:, :], [[[pred_scaled]]], axis=1)

    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": future_preds
    }).set_index("date")

    return {
        "test_result": {
            "dates": test_dates.strftime("%Y-%m-%d").tolist(),
            "y_pred": y_pred.tolist(),
            "y_true": y_true.tolist(),
            "metrics": {
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4),
            }
        },
        "future": forecast_df["forecast"]
    }
