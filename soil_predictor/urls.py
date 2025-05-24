from django.urls import path
from . import views

app_name = 'soil_predictor'

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/live/', views.predict_live, name='predict_live'),
    # Admin utilities
    path('admin/models/', views.admin_models, name='admin_models'),
    path('admin/model/convert/<str:model_name>/', views.convert_model, name='convert_model'),
    path('admin/model/info/', views.model_info, name='model_info'),
] 