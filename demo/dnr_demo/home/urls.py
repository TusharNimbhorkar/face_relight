'''
Created on 13 Jul 2019

@author: morris
'''
from django.conf.urls import url
from django.urls import path

from . import views

app_name = "home"

urlpatterns = [
    # url(r'^admin_panel$', views.admin_panel, name="admin_panel"),
    path('create-sh-previews', views.create_sh_previews),
    path('create-sh-presets', views.create_sh_presets),
    url(r'^$', views.home, name="home"),
]
