'''
Created on 13 Jul 2019

@author: morris
'''
from django.conf.urls import url

from . import views

app_name = "home"

urlpatterns = [
    # url(r'^admin_panel$', views.admin_panel, name="admin_panel"),
    url(r'^$', views.home, name="home"),
]
