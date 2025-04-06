import tensorflow as tf
from pathlib import Path

# Список валют
currencies = ['CNY', 'CZK', 'GBP', 'HUF', 'PLN', 'USD']

# Базовый путь к моделям
base_path = Path('rates/forecasting/trained_models')

for currency in currencies:
    h5_model_path = base_path / f'lstm_{currency}.h5'
    tflite_model_path = base_path / f'lstm_{currency}.tflite'

    print(f'Конвертируем модель для {currency}...')

    # Загружаем модель
    model = tf.keras.models.load_model(h5_model_path)

    # Создаём конвертер
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # ВАЖНО: добавляем поддержку select TF ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # базовые операции tflite
        tf.lite.OpsSet.SELECT_TF_OPS    # расширенные операции TensorFlow
    ]

    # ВАЖНО: отключаем экспериментальное упрощение tensor list ops
    converter._experimental_lower_tensor_list_ops = False

    # Опциональная оптимизация для уменьшения размера модели
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Конвертация модели
    tflite_model = converter.convert()

    # Сохраняем результат
    tflite_model_path.write_bytes(tflite_model)

    print(f'✅ Модель {currency} успешно сконвертирована в {tflite_model_path}')

print("🎉 Все модели успешно конвертированы в TensorFlow Lite!")
