import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta
from rates.models import ExchangeRateNormalized

def predict_linear_regression(currency, days=30, test_size=0.2):
    """
    Классическая линейная регрессия: прогноз курса валюты по дате (ordinal).
    """

    # Загружаем данные
    df = pd.DataFrame(list(ExchangeRateNormalized.objects.filter(
        currency__currency_code=currency).order_by("date").values()))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by="date").reset_index(drop=True)

    # Преобразуем дату в числовой формат
    df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)
    X_all = df[['date_ordinal']]
    y_all = df['rate_value'].astype(float)
    dates_all = df['date']

    # Делим на train/test
    split_index = int(len(df) * (1 - test_size))
    X_train = X_all.iloc[:split_index]
    y_train = y_all.iloc[:split_index]
    X_test = X_all.iloc[split_index:]
    y_test = y_all.iloc[split_index:]
    test_dates = dates_all.iloc[split_index:]

    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Предсказание на тесте
    y_pred_test = model.predict(X_test_scaled)
    metrics = {
        "MSE": float(mean_squared_error(y_test, y_pred_test)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "MAE": float(mean_absolute_error(y_test, y_pred_test)),
        "R2": float(r2_score(y_test, y_pred_test)),
    }

    print(f"\n[DEBUG] Тестовые метрики:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Прогноз на будущее
    last_date = df['date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_ordinal = pd.Series(future_dates).map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    future_scaled = scaler.transform(future_ordinal)
    future_preds = model.predict(future_scaled)

    print(f"\n[DEBUG] Первые 5 прогнозов на будущее: {future_preds[:5]}")

    return {
        "future": pd.Series(future_preds, index=future_dates),
        "test_result": {
            "dates": test_dates.dt.strftime('%Y-%m-%d').tolist(),
            "y_true": y_test.tolist(),
            "y_pred": y_pred_test.tolist(),
            "metrics": metrics,
        },
    }
