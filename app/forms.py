from typing import Text
from django import forms
from django.db.models.enums import IntegerChoices
from django.forms import ModelForm
from django.forms.fields import CharField, ChoiceField, IntegerField
from django.forms.models import ModelChoiceField, ModelMultipleChoiceField
from django.forms.widgets import PasswordInput, TextInput
from pandas.core import api
from app.models import Pacientes, Profissional, Consulta
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
            'nome': forms.TextInput(attrs={'id':'nome','placeholder': 'Insira o Nome', 'class': 'text-left'}),
            'dataNasc': forms.DateInput(attrs={'id':'dataNasc','placeholder': 'Insira a Data de Nascimento'}),
            'altura': forms.NumberInput(attrs={'id':'altura','placeholder': 'Insira a Altura'}),
            'cx': forms.TextInput(attrs={'id':'cx','placeholder': 'Insira o Tipo da Cirurgia'}),
            'data': forms.DateInput(attrs={'id':'data','placeholder': 'Insira a Data da Cirurgia'}),
            'alta': forms.NumberInput(attrs={'id':'alta','placeholder': 'Insira a Alta'}),
            'pesoPreOp': forms.NumberInput(attrs={'id':'pesoPreOp','placeholder': 'Insira o Peso Pré-Operatório'}),
        }
        labels = {
            'pesoPreOp': 'Peso pré-operatório'
        }

    sexo = ChoiceField(choices=SEXO, widget=forms.Select(attrs={'id': 'sexo'}))
    proficional = CustomMMCF(queryset=Profissional.objects.all(),
            widget=forms.CheckboxSelectMultiple(attrs={'id':'proficional'}))

class LoginForm(forms.Form):
    username = CharField(required=True, widget=TextInput(attrs={'placeholder': 'Insira seu Login', 'id':'usuario'}), label="Usuário")
    password = CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Insira sua Senha', 'id':'senha'}), required=True, label="Senha")

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

class ConsultaForm(ModelForm):
  
    class Meta:
        model = Consulta
        fields = '__all__'
        widgets = {
            'peso': forms.NumberInput(attrs={'id':'peso','placeholder': 'Insira o Peso'}),
            'ca': forms.NumberInput(attrs={'id':'ca','placeholder': 'Insira Circunferência Abdominal'}),
            'rcq': forms.NumberInput(attrs={'id':'rcq','placeholder': 'Insira a Relação Cintura Quadril'}),
            'gc': forms.NumberInput(attrs={'id':'gc','placeholder': 'Insira a Gordura Corporal'}),
            'data': forms.DateInput(attrs={'id':'data','placeholder': 'Insira a Data'}),
        }