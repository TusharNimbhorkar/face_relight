#!/usr/bin/python
# Here we can specify parameters for the celery workers
from .settings import * 
DEBUG = False
CELERYD_PREFETCH_MULTIPLIER = 1
CELERY_ACKS_LATE = 1
IS_CELERY_WORKER = True

