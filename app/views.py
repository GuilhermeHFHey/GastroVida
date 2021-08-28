from django.shortcuts import render, redirect
from app.forms import PacientesForm
from app.models import Pacientes
from django.template.loader import render_to_string
from django.http import JsonResponse
from django.contrib import messages
from tablib import Dataset
from django.core.paginator import Paginator
# Create your views here.


def home(request):
    data = {}
    search = request.GET.get("q")
    if search:
        pacientes = Pacientes.objects.filter(nome__icontains=search)
    else:
        pacientes = Pacientes.objects.all()
    paginator = Paginator(pacientes, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    data['db'] = page_obj
    if request.is_ajax():
        html = render_to_string(
            template_name="index_partial.html",
            context={'db': pacientes}
        )
        data_dict = {"html_from_view": html}
        return JsonResponse(data=data_dict, safe=False)

    return render(request, 'index.html', data)


def form(request):
    data = {}
    data['form'] = PacientesForm()
    return render(request, 'form.html', data)


def create(request):
    form = PacientesForm(request.POST or None)
    if form.is_valid():
        form.save()
        return redirect('home')


def view(request, pk):
    data = {}
    data['db'] = Pacientes.objects.get(pk=pk)
    return render(request, 'view.html', data)


def edit(request, pk):
    data = {}
    data['db'] = Pacientes.objects.get(pk=pk)
    data['form'] = PacientesForm(instance=data['db'])
    return render(request, 'form.html', data)


def update(request, pk):
    data = {}
    data['db'] = Pacientes.objects.get(pk=pk)
    form = PacientesForm(request.POST or None, instance=data['db'])
    if form.is_valid():
        form.save()
        return redirect('home')


def delete(request, pk):
    db = Pacientes.objects.get(pk=pk)
    db.delete()
    return redirect('home')


def uploadExel(request):
    if request.method == 'POST':
        dataset = Dataset()
        dados = request.FILES['myfile']
        if not dados.name.endswith('xlsx'):
            messages.info(request, 'Formato Incorreto')
            return render(request, 'import.html')

        dadosImportados = dataset.load(dados.read(), format='xlsx')
        for paciente in dadosImportados:
            value = Pacientes(
                paciente[0],
                paciente[0],
                paciente[1],
                paciente[2],
                paciente[3],
                paciente[4],
                paciente[5],
                paciente[6],
                paciente[7],
                paciente[8],
                paciente[9],
                paciente[10],
                paciente[11],
                paciente[12],
                paciente[13],
                paciente[14],
                paciente[15],
                paciente[16],
                paciente[17],
                paciente[18],
                paciente[19],
                paciente[20]
            )
            value.save()
    return render(request, 'import.html')
