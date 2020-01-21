from django import forms
from .models import Uploads


class UploadForm(forms.ModelForm):
    class Meta:
        model = Uploads
        fields = ('file',)