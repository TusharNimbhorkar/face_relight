'''
Created on 6 Apr 2016

@author: morris
'''
import os
from celery import Celery

# Use a custom settings file for Celery workers
this_module = os.path.basename(os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', '{}.celery_settings'.format(this_module))

from django.conf import settings  # import the custom celery settings

app = Celery(this_module, backend='redis', broker='redis://localhost:6379/1')
app.config_from_object(settings, namespace='CELERY')
app.autodiscover_tasks()
