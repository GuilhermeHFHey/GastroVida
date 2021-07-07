from django.shortcuts import render, redirect
from app.forms import PacientesForm
from app.models import Pacientes
from django.template.loader import render_to_string
from django.http import JsonResponse
# Create your views here.


def home(request):
    data = {}
    search = request.GET.get("q")
    if search:
        pacientes = Pacientes.objects.filter(nome__icontains=search)
    else:
        pacientes = Pacientes.objects.all()
    data['db'] = pacientes
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
