from typing import Text
from django import forms
from django.db.models.enums import IntegerChoices
from django.forms import ModelForm
from django.forms.fields import CharField, ChoiceField, IntegerField
from django.forms.models import ModelChoiceField, ModelMultipleChoiceField
from django.forms.widgets import PasswordInput, TextInput
from pandas.core import api
from app.models import Pacientes, Profissional
from django.contrib.auth import get_user_model

# Create the form class.

class CustomMMCF(forms.ModelMultipleChoiceField):
    def label_from_instance(self, member):
        return "%s" % member.nome

class PacientesForm(ModelForm):
    SEXO = (
        ('M', 'Masculino'),
        ('F', 'Feminino'),
    )    
    class Meta:
        model = Pacientes
        fields = '__all__'
        widgets = {
            'nome': forms.TextInput(attrs={'placeholder': 'Insira o Nome', 'class': 'text-left'}),
            'idade': forms.TextInput(attrs={'placeholder': 'Insira a Idade'}),
            'sexo': forms.TextInput(attrs={'placeholder': 'Insira o Sexo'}),
            'imc': forms.TextInput(attrs={'placeholder': 'Insira o IMC'}),
            'altura': forms.TextInput(attrs={'placeholder': 'Insira a Altura'}),
            'ca': forms.TextInput(attrs={'placeholder': 'Insira a Circunferência Abdominal'}),
            'rcq': forms.TextInput(attrs={'placeholder': 'Insira a Relação Cintura Quadril'}),
            'gc': forms.TextInput(attrs={'placeholder': 'Insira a Gordura Corporal'}),
            'cx': forms.TextInput(attrs={'placeholder': 'Insira o Tipo da Cirurgia'}),
            'data': forms.TextInput(attrs={'placeholder': 'Insira a Data da Cirurgia'}),
            'alta': forms.TextInput(attrs={'placeholder': 'Insira a Alta'}),
            'tpo': forms.TextInput(attrs={'placeholder': 'Insira o Tempo Pós Operatorio'}),
            'mes1': forms.TextInput(attrs={'placeholder': 'Insira a Perda de Peso 1ºMês'}),
            'mes3': forms.TextInput(attrs={'placeholder': 'Insira a Perda de Peso 3ºMês'}),
            'mes6': forms.TextInput(attrs={'placeholder': 'Insira a Perda de Peso 6ºMês'}),
            'mes9': forms.TextInput(attrs={'placeholder': 'Insira a Perda de Peso 9ºMês'}),
            'ano1': forms.TextInput(attrs={'placeholder': 'Insira a Perda de Peso 1ºAno'}),
            'ano2': forms.TextInput(attrs={'placeholder': 'Insira a Perda de Peso 2ºAno'}),
            'ano3': forms.TextInput(attrs={'placeholder': 'Insira a Perda de Peso 3ºAno'}),
            'ano4': forms.TextInput(attrs={'placeholder': 'Insira a Perda de Peso 4ºAno'}),
            'ano5': forms.TextInput(attrs={'placeholder': 'Insira a Perda de Peso 5ºAno'}),
        }

    sexo = ChoiceField(choices=SEXO)
    proficional = CustomMMCF(queryset=Profissional.objects.all(),
            widget=forms.CheckboxSelectMultiple)

class LoginForm(forms.Form):
    username = CharField(required=True, widget=TextInput(attrs={'placeholder': 'Insira seu Login'}))
    password = CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Insira sua Senha'}), required=True)

    def cleaned(self):
        data = {}
        data['login'] = self.cleaned_data['login']
        data['senha'] = self.cleaned_data['senha']
        return data

user = get_user_model()
class RegisterForm(forms.Form):
    nome = forms.CharField(max_length=150)
    prof = forms.CharField(max_length=150)
    username = forms.CharField(max_length=150)
    password = forms.CharField(max_length=150)
