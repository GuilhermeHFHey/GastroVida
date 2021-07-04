from django.shortcuts import render, redirect
from app.forms import PacientesForm
from app.models import Pacientes

# Create your views here.
def home(request):
    data = {}
    data['db'] = Pacientes.objects.all()
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

def update(request,pk):
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