from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.challenge1_view, name='home'),
    path('challenge1/', views.challenge1_view, name='challenge1_view'),
    path('challenge2/', views.challenge2_view, name='challenge2_view'),
    path('challenge3/', views.challenge3_view, name='challenge3_view'),
    path('challenge4/', views.challenge4_view, name='challenge4_view'),
    path('challenge5/', views.challenge5_view, name='challenge5_view'),
    path('challenge6/', views.challenge6_view, name='challenge6_view'),
    path('challenge7/', views.challenge7_view, name='challenge7_view'),
    path('challenge8/', views.challenge8_view, name='challenge8_view'),
    path('challenge9/', views.challenge9_view, name='challenge9_view'),
    path('extension1/', views.extension1_view, name='extension1_view'),
    path('extension2/', views.extension2_view, name='extension2_view'),
]   