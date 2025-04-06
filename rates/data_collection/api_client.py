import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

ECB_API_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"

# Список официальных праздников (нужно обновлять для каждого года)
HOLIDAYS = {
    2024: [
        "2024-01-01",  # Новый год
        "2024-03-29",  # Великая Пятница (Good Friday)
        "2024-04-01",  # Пасхальный понедельник (Easter Monday)
        "2024-05-01",  # День труда
        "2024-12-25",  # Рождество
        "2024-12-26",  # Второй день Рождества
    ],
    2025: [
        "2025-01-01",
        "2025-04-18",
        "2025-04-21",
        "2025-05-01",
        "2025-12-25",
        "2025-12-26",
    ],
}


def get_last_working_day():
    """Определяет последний рабочий день (учитывая выходные и праздники)."""
    today = datetime.today().date()

    # Если сегодня выходной или праздник, ищем последний рабочий день
    while today.weekday() in [5, 6] or today.strftime("%Y-%m-%d") in HOLIDAYS.get(today.year, []):
        today -= timedelta(days=1)

    return today


def fetch_exchange_rates():
    """
    Получает данные с API Европейского центрального банка.
    Если данных за сегодня нет, всё равно создаём запись с текущей датой.
    """
    today = datetime.today().date()  # Берём текущую дату

    response = requests.get(ECB_API_URL)
    if response.status_code != 200:
        raise Exception(f"Ошибка при запросе API: {response.status_code}")

    # Парсим XML
    root = ET.fromstring(response.content)
    namespaces = {'default': 'http://www.ecb.int/vocabulary/2002-08-01/eurofxref'}

    # Ищем курсы за последнюю доступную дату
    date_node = root.find(".//default:Cube[@time]", namespaces=namespaces)
    if date_node is None or 'time' not in date_node.attrib:
        raise Exception("Не удалось найти дату в XML.")

    ecb_date = datetime.strptime(date_node.attrib['time'], "%Y-%m-%d").date()

    # Сохраняем курсы валют
    rates = {}
    for rate_node in date_node.findall("default:Cube", namespaces=namespaces):
        currency = rate_node.attrib.get('currency')
        rate = rate_node.attrib.get('rate')
        if currency and rate:
            rates[currency] = float(rate)

    # ✅ Всегда возвращаем **текущую дату**, даже если данные не обновились
    return today, rates
