from django.shortcuts import render
from django.http import HttpResponseBadRequest, JsonResponse, HttpResponseNotFound
import django_tables2 as tables

from rest_framework.decorators import api_view
from rest_framework.decorators import parser_classes
from rest_framework.parsers import JSONParser

from .forms import UploadForm
from .dnr import process_image, process_image_sh_mul
from . import models
from django.conf import settings
import os 
import humanhash
import imutils
import numpy as np
import cv2
import shutil

def handle_uploaded_file(f):
    file_id = humanhash.uuid()[0]
    filename = "file_%s.jpg" % file_id
    filepath = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
    os.makedirs(os.path.join(settings.MEDIA_ROOT, 'uploads'), exist_ok=True)
    
    with open(filepath, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

    # img = np.array(cv2.imread(filepath))
    #
    # if img.shape[0]<img.shape[1]:
    #     img = imutils.resize(img, width=1024)
    # else:
    #     img = imutils.resize(img, width=1024)

    # cv2.imwrite(filepath, img)
    return filepath, file_id


# '/'
def home(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)

        if form.is_valid():
            filepath, file_id = handle_uploaded_file(request.FILES['file'])
            result, is_face_found = process_image(filepath, upload_id=file_id)

            return JsonResponse({"upload_id": file_id, "is_face_found":is_face_found})

        return HttpResponseBadRequest()
    else:
        form = UploadForm()
        context = {"form" : form}
        return render(request, 'index.html', context)

# '/create-sh-previews'
@api_view(['POST'])
@parser_classes([JSONParser])
def create_sh_previews(request):
    if request.method == 'POST':
        print(request.data)

        file_id = request.data.get('file_id')
        preset_name = request.data.get('preset_name')
        sh_id = request.data.get('sh_id')

        filename = "file_%s.jpg" % file_id
        filepath = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)

        if filename.startswith('file_sample_'):
            filepath = os.path.join(settings.MEDIA_ROOT, 'output', file_id, 'ori.jpg')

        result, is_face_found = process_image_sh_mul(filepath, preset_name, sh_id, upload_id=file_id)

        return JsonResponse({"upload_id": file_id, "is_face_found":is_face_found})

    return HttpResponseNotFound('<h1>Page not found</h1>')

# '/create-sh-presets'
@api_view(['POST'])
@parser_classes([JSONParser])
def create_sh_presets(request):
    if request.method == 'POST':
        file_id = request.data.get('file_id')
        sh_mul = request.data.get('sh_mul')

        filename = "file_%s.jpg" % file_id
        filepath = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)

        if filename.startswith('file_sample_'):
            filepath = os.path.join(settings.MEDIA_ROOT, 'output', file_id, 'ori.jpg')

        result, is_face_found = process_image(filepath, sh_mul, upload_id=file_id)

        return JsonResponse({"upload_id": file_id, "is_face_found":is_face_found})

    return HttpResponseNotFound('<h1>Page not found</h1>')
    
# '/regenerate-gallery'
def regenerate_gallery(request):
    original_gallery_dir = os.path.join(settings.MEDIA_ROOT, 'gallery_images')
    output_dir = os.path.join(settings.MEDIA_ROOT, 'output')

    for filename in os.listdir(original_gallery_dir):
        filepath = os.path.join(original_gallery_dir, filename)
        if os.path.isfile(filepath):
            image_id = os.path.splitext(filename)[0]

            if os.path.exists(os.path.join(output_dir, image_id)):
                shutil.rmtree(os.path.join(output_dir, image_id))

            process_image(filepath, upload_id=image_id)
    
    return JsonResponse({"done": True})