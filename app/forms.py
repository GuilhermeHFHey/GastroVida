from typing import Text
from django import forms
from django.db.models.enums import IntegerChoices
from django.forms import ModelForm
from django.forms.fields import CharField, ChoiceField, IntegerField
from django.forms.forms import Form
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
        }
        labels = {
            'dataNasc':'Data de Nascimento',
        }

    sexo = ChoiceField(choices=SEXO, widget=forms.Select(attrs={'id': 'sexo'}))
    medico = CustomMMCF(queryset=Profissional.objects.filter(prof='Médico'),
            widget=forms.CheckboxSelectMultiple(attrs={'name':'proficional'}))
    nutricionista = CustomMMCF(queryset=Profissional.objects.filter(prof='Nutricionista'),
            widget=forms.CheckboxSelectMultiple(attrs={'name':'proficional'}))
    cardiologista = CustomMMCF(queryset=Profissional.objects.filter(prof='Cardiologista'),
            widget=forms.CheckboxSelectMultiple(attrs={'name':'proficional'}))
    psicologo = CustomMMCF(queryset=Profissional.objects.filter(prof='Psicologo'),
            widget=forms.CheckboxSelectMultiple(attrs={'name':'proficional'}))
    educadorFisico = CustomMMCF(queryset=Profissional.objects.filter(prof='EducadorFisico'),
            widget=forms.CheckboxSelectMultiple(attrs={'name':'proficional'}))

    def __init__(self, *args, **kwargs):
        super(PacientesForm, self).__init__(*args, **kwargs)
        self.fields['educadorFisico'].label = "Educador Fisico"

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
    PROFISSAO = (
        ('Médico', 'Médico'),
        ('Nutricionista', 'Nutricionista'),
        ('Cardiologista', 'Cardiologista'),
        ('Psicologo', 'Psicologo'),
        ('EducadorFisico', 'Educador Fisico'),
        ('Secretaria', 'Secretaria'),
    )
    nome = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Insira o Nome do Profissional', 'id':'nome'}), max_length=150, label='Nome do Profissional')
    prof = ChoiceField(label='Profissão', choices=PROFISSAO, widget=forms.Select(attrs={'id': 'profissao'}))
    username = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Insira um Usuario para Login', 'id':'username'}), max_length=150, label='Usuario')
    password = CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Insira a Senha para Login', 'id':'password'}), required=True, label="Senha")


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
            'cx': forms.TextInput(attrs={'id':'cx','placeholder': 'Insira o Tipo da Cirurgia'}),
            'alta': forms.NumberInput(attrs={'id':'alta','placeholder': 'Insira a Alta (Dias)'}),
            'pesoPreOp': forms.NumberInput(attrs={'id':'pesoPreOp','placeholder': 'Insira o Peso Pré-Operatório'})
        }
        labels = {
            'data': 'Data',
            'ca':'Circunferência Abdominal',
            'rcq': 'Relação Cintura Quadril',
            'gc': 'Gordura Corporal (%)',
            'cx': 'Tipo da Cirurgia',
            'alta': 'Alta',
            'pesoPreOp': 'Peso Pré-Operatório'
        }
