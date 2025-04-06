# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models

class ExchangeRate(models.Model):
    id = models.AutoField(primary_key=True)  # ID, уже AUTO_INCREMENT
    date = models.DateField()  # Поле "date"
    USD = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    CNY = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    HUF = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    PLN = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    CZK = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    GBP = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)

    class Meta:
        db_table = 'exchange_rates'  # Имя существующей таблицы


class ExchangeRateTimeSeries(models.Model):
    date = models.DateField(primary_key=True)  # Используем date как первичный ключ
    usd_eur = models.DecimalField(max_digits=10, decimal_places=6)
    cny_eur = models.DecimalField(max_digits=10, decimal_places=6)
    huf_eur = models.DecimalField(max_digits=10, decimal_places=6)
    pln_eur = models.DecimalField(max_digits=10, decimal_places=6)
    czk_eur = models.DecimalField(max_digits=10, decimal_places=6)
    gbp_eur = models.DecimalField(max_digits=10, decimal_places=6)

    class Meta:
        db_table = 'exchange_rates_time_series'  # Название таблицы в БД
        managed = False  # Django не будет управлять этой таблицей


# Объединённое определение модели Currency
class Currency(models.Model):
    currency_code = models.CharField(max_length=3, primary_key=True)
    currency_name = models.CharField(max_length=50)

    class Meta:
        db_table = 'currencies'


# Таблица для нормализованных курсов
class ExchangeRateNormalized(models.Model):
    date = models.DateField()
    # Указываем внешнюю связь на единственную модель Currency
    currency = models.ForeignKey(Currency, on_delete=models.CASCADE, db_column="currency_code")
    rate_value = models.DecimalField(max_digits=10, decimal_places=4)

    class Meta:
        db_table = 'exchange_rates_normalized'
        unique_together = ('date', 'currency')


# Предсказания
class Prediction(models.Model):
    date = models.DateField()
    currency_code = models.ForeignKey(Currency, on_delete=models.CASCADE, db_column="currency_code")
    predicted_value = models.DecimalField(max_digits=10, decimal_places=4)
    model_name = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'predictions'

