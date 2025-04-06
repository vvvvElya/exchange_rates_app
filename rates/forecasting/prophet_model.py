from prophet import Prophet
import pandas as pd
from rates.models import ExchangeRateNormalized
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
    }

def predict_prophet(currency, forecast_steps):
    data = ExchangeRateNormalized.objects.filter(currency__currency_code=currency).order_by('date')
    if not data.exists():
        raise ValueError(f"Нет данных для валюты {currency}")

    df = pd.DataFrame(list(data.values('date', 'rate_value')))
    df = df.rename(columns={'date': 'ds', 'rate_value': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')

    # 🔥 Исправляем разбиение, чтобы даты были без пробелов
    split_date = df['ds'].iloc[int(len(df)*0.9) - 1]  # последняя дата train
    train_df = df[df['ds'] <= split_date]
    test_df = df[df['ds'] > split_date]

    # Праздники (твои)
    holidays = pd.DataFrame({
        'holiday': 'holiday',
        'ds': pd.to_datetime([
            '2024-01-01', '2024-04-01', '2024-05-01', '2024-05-08',
            '2024-07-05', '2024-08-29', '2024-09-01', '2024-09-15',
            '2024-11-01', '2024-12-24', '2024-12-25', '2024-12-26',
            '2025-01-01'
        ]),
        'lower_window': 0,
        'upper_window': 0
    })

    # Prophet модель
    model = Prophet(
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(train_df)

    # 🔥 Важно! прогноз на тестовых датах из базы
    forecast_test = model.predict(test_df[['ds']])

    # прогноз на будущее
    last_date = df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
    forecast_future = model.predict(pd.DataFrame({'ds': future_dates}))

    # считаем метрики
    metrics = calculate_metrics(test_df['y'].values, forecast_test['yhat'].values)

    result = {
        "test_result": {
            "dates": test_df['ds'].dt.strftime('%Y-%m-%d').tolist(),
            "y_true": test_df['y'].tolist(),
            "y_pred": forecast_test['yhat'].tolist(),
            "metrics": metrics,
        },
        "future": forecast_future.set_index('ds')['yhat'],
        "future_lower": forecast_future.set_index('ds')['yhat_lower'],
        "future_upper": forecast_future.set_index('ds')['yhat_upper'],
    }

    return result
