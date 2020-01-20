from django.db import models
from django import forms


# class Uploads(models.Model):
#     file            = models.FileField(upload_to='uploads/')
#     created_at      = models.DateTimeField(auto_now_add=True)
#     classification  = models.CharField(max_length=50, null=True, default=None)

class Uploads(models.Model):
    theta           = models.FloatField(default=0, max_length=5)
    phi           = models.FloatField(default=0, max_length=5)
    r           = models.FloatField(default=30, max_length=5)
    isPreset = models.BooleanField(default=False)
    presetName = models.CharField(default="zalando", max_length=10)
