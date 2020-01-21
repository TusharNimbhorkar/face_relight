from django.shortcuts import render
from django.http import HttpResponseBadRequest, JsonResponse
import django_tables2 as tables

from .forms import UploadForm
from .dnr import process_image
from . import models
from django.conf import settings
import os 
import humanhash

def handle_uploaded_file(f):
    filename = "file_%s.jpg" % humanhash.uuid()[0] 
    filepath = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
    os.makedirs(os.path.join(settings.MEDIA_ROOT, 'uploads'), exist_ok=True)
    
    with open(filepath, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return filepath

# '/'
def home(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)

        if form.is_valid():
            filepath = handle_uploaded_file(request.FILES['file'])
            result, is_face_found = process_image(filepath)

            return JsonResponse({"filename": result, "is_face_found":is_face_found})

        return HttpResponseBadRequest()
    else:
        form = UploadForm()
        context = {"form" : form}
        return render(request, 'index.html', context)