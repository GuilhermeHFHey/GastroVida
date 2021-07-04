from django.shortcuts import render
from app.forms import PacientesForm

# Create your views here.
def home(request):
    return render(request, 'index.html')

def form(request):
    data = {}
    data['form'] = PacientesForm()
    return render(request, 'form.html', data)