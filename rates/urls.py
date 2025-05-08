from django.urls import path
from . import views  # Импортируем views из текущего приложения
from .views import predictions_view
from django.conf import settings
from django.conf.urls.static import static




urlpatterns = [
    path('', views.home, name='home'),
    # path('graph/', views.graph_view, name='graph'),  # Удалил дубликат
    path("graph/", views.graph_view, name="exchange_rate_graph"),  #  Путь для графика
    path('predikcia/', predictions_view, name='predikcia'),
    path('backfill/', views.backfill_view, name='backfill'),
    path('kontakt/', views.contact_view, name='kontakt'),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

