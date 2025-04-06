from pathlib import Path

# Укажи свои значения здесь
database_url = "postgresql://currency_db_6yye_user:HXFB8MfWU7I1H3TcA6UF81zjuHQ1qEuv@dpg-cvp8moi4d50c73bq4fqg-a/currency_db_6yye"
secret_key = "django-insecure-=(o(@g^r_)+6p1tk=i8jo8&mf0%88s#uqc)4he50x1ydh6hwi+"
debug = "True"

# Создаём содержимое .env файла
env_content = f"""DATABASE_URL={database_url}
DJANGO_SECRET_KEY={secret_key}
DEBUG={debug}
"""

# Путь к .env файлу (проверь, что это корень твоего проекта)
env_path = Path(".env")

# Записываем файл с явной кодировкой UTF-8 без BOM
env_path.write_text(env_content, encoding="utf-8")

print(f".env файл успешно создан по пути: {env_path.resolve()}")
