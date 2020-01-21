from django.db import models
from django import forms


class Uploads(models.Model):
    file = models.FileField(upload_to='uploads/')