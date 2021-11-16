"""projeto URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from os import name
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as authViews
# from app.views import home, form, create, view, edit, update, delete, uploadExel, loginPage, registerPage, initial, PrevPred

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('', home, name='home'),
    # path('form/', form, name='form'),
    # path('create/', create, name='create'),
    # path('view/<int:pk>/', view, name='view'),
    # path('edit/<int:pk>/', edit, name='edit'),
    # path('update/<int:pk>/', update, name='update'),
    # path('delete/<int:pk>/', delete, name='delete'),
    # path('upload/', uploadExel, name='uploadExel'),
    # path('PrevPred/<int:pk>/', PrevPred, name='PrevPred'),
    # path('login/', loginPage, name='login'),
    # path('register/', registerPage, name='register'),
    # path('inicio/', initial, name='initial'),
]
