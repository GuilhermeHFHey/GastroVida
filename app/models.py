from django.db import models
from django.forms.fields import ChoiceField
from django.contrib.auth.models import AbstractUser
from django_cryptography.fields import encrypt, get_encrypted_field
from numpy import mod
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
# # Create your models here.


class Profissional(AbstractUser):
    nome = encrypt(models.CharField(max_length=150, editable=False))
    prof = encrypt(models.CharField(max_length=150, editable=False))

class Pacientes(models.Model):
    SEXO = (
        ('M', 'Masculino'),
        ('F', 'Feminino'),
    )
    id = models.BigAutoField(primary_key=True)
    nome = encrypt(models.CharField(max_length=150, default=''))
    dataNasc = models.DateField(default='')
    sexo = encrypt(models.CharField(max_length=1, choices=SEXO, default=''))
    altura = encrypt(models.FloatField(max_length=30, default=''))
    cx = encrypt(models.CharField(max_length=100, default=''))
    data = encrypt(models.DateField(default=''))
    alta = models.DecimalField(decimal_places=0, max_digits=3, default='')
    pesoPreOp = models.FloatField(default='')
    proficional = models.ManyToManyField(Profissional)

    def getIdade(self):
        return relativedelta(date.today(), self.dataNasc).years

    def getIMC(self):
        return round(self.pesoPreOp/(self.altura*self.altura), 2)

    def getPesoIdeal(self):
        if self.sexo == 'M':
            return round(61.2328 + ((self.altura - 1.524)*53.5433), 2)
        else:
            return round(53.975 + ((self.altura - 1.524)*53.5433), 2)

    def getExcesso(self):
        return round(self.pesoPreOp - self.getPesoIdeal(), 2)


class Consulta(models.Model):
    id = models.BigAutoField(primary_key=True)
    numConsulta = models.FloatField(default='')
    peso = models.FloatField(default='',  max_length=5)
    ca = encrypt(models.DecimalField(default='',  decimal_places=0, max_digits=3))  # circunferencia abdominal
    rcq = encrypt(models.DecimalField(default='', decimal_places=0, max_digits=3))  # relacao cintura quadril
    gc = encrypt(models.FloatField(default='', max_length=5))
    data = encrypt(models.DateField(default=''))
    paciente = models.ForeignKey(Pacientes, on_delete=models.CASCADE)

    def getIMC(self):
        return round(self.peso/(self.paciente.altura*self.paciente.altura), 2)

    def getPerda(self):
        return round(self.paciente.pesoPreOp - self.peso, 2)

    def getPerdaPerc(self):
        return round(100 * (self.getPerda()/self.paciente.getExcesso()), 2)


    
