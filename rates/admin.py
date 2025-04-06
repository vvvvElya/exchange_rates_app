from django.contrib import admin
from .models import ExchangeRate
from .models import Currency
admin.site.register(Currency)
admin.site.register(ExchangeRate)

from .models import ExchangeRateNormalized

admin.site.register(ExchangeRateNormalized)

