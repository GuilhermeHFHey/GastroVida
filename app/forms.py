from django import forms
from app.models import Pacientes

# Create the form class.


class PacientesForm(forms.Form):
    nome = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Insira o Nome'}))
    idade = forms.IntegerField()
    sexo = forms.CharField()
    imc = forms.FloatField()
    altura = forms.FloatField()
    ca = forms.IntegerField()  # circunferencia abdominal
    rcq = forms.IntegerField()  # relacao cintura quadril
    gc = forms.FloatField()
    cx = forms.CharField()
    data = forms.DateField()
    alta = forms.IntegerField()
    tpo = forms.IntegerField()
    mes1 = forms.FloatField()
    mes3 = forms.FloatField()
    mes6 = forms.FloatField()
    mes9 = forms.FloatField()
    ano1 = forms.FloatField()
    ano2 = forms.FloatField()
    ano3 = forms.FloatField()
    ano4 = forms.FloatField()
    ano5 = forms.FloatField()
