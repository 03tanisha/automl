from django.urls import path
from . import  views

urlpatterns = [
    path('', views.home, name='home'),      
    path('upload/', views.home, name='upload'),
    path('automl-service/', views.predict, name='automl-service'),
]
