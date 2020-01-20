from django.shortcuts import render
from django.http import HttpResponseBadRequest, JsonResponse
import django_tables2 as tables

from .forms import UploadForm
from .dnr import process_image
from . import models


# '/'
def home(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            upload = form.instance  # from .models import Uploads

            if upload.isPreset:

                filenames = []

                if upload.presetName == 'zalando':
                    filenames.append(process_image(90, 90, 30))
                    filenames.append(process_image(80, 130, 30))
                    filenames.append(process_image(0, -3, 28))
                    filenames.append(process_image(80, 10, 28))
                    filenames.append(process_image(80, 150, 14))
                elif upload.presetName == 'bol.com':
                    filenames.append(process_image(85, 130, 30))
                    filenames.append(process_image(90, 90, 30))
                    filenames.append(process_image(90, 270, 30))
                    filenames.append(process_image(75, 0, 25))
                    filenames.append(process_image(85, 210, 30))
                    filenames.append(process_image(80, 240, 30))

                urls = []
                for fname in filenames:
                    urls.append("media/" + fname)

                return JsonResponse({"filename": urls})
            else:
                filename = process_image(upload.theta, upload.phi, upload.r)

                return JsonResponse({"filename": "media/"+filename})
        return HttpResponseBadRequest()
    else:
        form = UploadForm()
        context = {"form" : form}
        return render(request, 'home.html', context)


class UploadTable(tables.Table):
    class Meta:
        model = models.Uploads
        template_name = 'django_tables2/bootstrap.html'
        attrs = {'class': 'paleblue'}


# '/admin_panel'
def admin_panel(request):
    table = UploadTable(models.Uploads.objects.all())
    tables.RequestConfig(request, paginate=True).configure(table)
    context = {"table" : table}
    return render(request, 'admin_panel.html', context)
