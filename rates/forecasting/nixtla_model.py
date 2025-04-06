def predict_timegpt(currency, forecast_steps):
    import os
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from nixtlats import NixtlaClient
    from rates.models import ExchangeRateNormalized

    # 🔑 Загружаем API-ключ из .env
    load_dotenv()
    timegpt_api_key = os.getenv('TIMEGPT_API_KEY')
    if not timegpt_api_key:
        raise ValueError("TIMEGPT_API_KEY nie je nastavený v .env súbore!")

    # 🧩 Загружаем данные из базы данных
    data = pd.DataFrame(list(ExchangeRateNormalized.objects.filter(currency__currency_code=currency).values()))
    if data.empty:
        raise ValueError(f'Nie sú dostupné historické údaje pre {currency}')

    if 'date' not in data.columns or 'rate_value' not in data.columns:
        raise ValueError(f'V dátach chýba požadovaný stĺpec "date" alebo "rate_value"')

    # 📈 Подготовка данных
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    df = data.rename(columns={'date': 'ds', 'rate_value': 'y'}).dropna()

    # 🔥 Разделение train/test
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    forecast_horizon_test = len(test_df)
    forecast_horizon_total = forecast_horizon_test + forecast_steps

    # 🧠 Инициализация клиента TimeGPT
    client = NixtlaClient(api_key=timegpt_api_key)

    # 🔮 Получение прогноза TimeGPT
    forecast = client.forecast(
        df=train_df,
        h=forecast_horizon_total,
        time_col='ds',
        target_col='y',
        finetune_steps=100,
        finetune_loss='default'
    )

    # 📏 Метрики на тестовой части
    forecast_test = forecast.iloc[:forecast_horizon_test]
    mae = mean_absolute_error(test_df['y'], forecast_test['TimeGPT'])
    rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_test['TimeGPT']))
    mse = mean_squared_error(test_df['y'], forecast_test['TimeGPT'])
    r2 = r2_score(test_df['y'], forecast_test['TimeGPT'])

    metrics = {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MSE': round(mse, 4),
        'R2': round(r2, 4)
    }

    # 📊 Подготовка результата в унифицированном формате
    result = {
        'test_result': {
            'dates': test_df['ds'].dt.strftime('%Y-%m-%d').tolist(),
            'y_true': test_df['y'].tolist(),
            'y_pred': forecast_test['TimeGPT'].tolist(),
            'metrics': metrics
        },
        'future': pd.Series(
            data=forecast['TimeGPT'].tolist()[forecast_horizon_test:],
            index=pd.to_datetime(forecast['ds'].tolist()[forecast_horizon_test:])
        )
    }

    return result
