from django.db import models
from django.forms.fields import ChoiceField
from django.contrib.auth.models import AbstractUser
from django_cryptography.fields import encrypt, get_encrypted_field
from numpy import mod
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from django.utils import timezone
# # Create your models here.


class Profissional(AbstractUser):
    PROFISSAO = (
        ('Médico', 'Médico'),
        ('Nutricionista', 'Nutricionista'),
        ('Cardiologista', 'Cardiologista'),
        ('Psicologo', 'Psicologo'),
        ('EducadorFisico', 'EducadorFisico'),
    )
    nome = encrypt(models.CharField(max_length=150, editable=False))
    prof = models.CharField(max_length=150, editable=False, choices=PROFISSAO)


class Pacientes(models.Model):
    SEXO = (
        ('M', 'Masculino'),
        ('F', 'Feminino'),
    )
    id = models.BigAutoField(primary_key=True)
    nome = models.CharField(max_length=150, default='')
    dataNasc = models.DateField(default='')
    sexo = encrypt(models.CharField(max_length=1, choices=SEXO, default=''))
    altura = encrypt(models.FloatField(max_length=30, default=''))
    proficional = models.ManyToManyField(Profissional)

    def getIdade(self):
        return relativedelta(date.today(), self.dataNasc).years

    def getPesoIdeal(self):
        if self.sexo == 'M':
            return round(61.2328 + ((self.altura - 1.524)*53.5433), 2)
        else:
            return round(53.975 + ((self.altura - 1.524)*53.5433), 2)



class Consulta(models.Model):
    EVENTOS = (
        ('Consulta', 'Consulta'),
        ('Cirurgia', 'Cirurgia')
    )
    id = models.BigAutoField(primary_key=True)
    numConsulta = models.FloatField(default=0)
    peso = models.FloatField(default=0,  max_length=5)
    ca = encrypt(models.DecimalField(default=0,  decimal_places=0, max_digits=3))  # circunferencia abdominal
    rcq = encrypt(models.DecimalField(default=0, decimal_places=0, max_digits=3))  # relacao cintura quadril
    gc = encrypt(models.FloatField(default=0, max_length=5))
    evento = models.CharField(max_length=8, choices=EVENTOS, default='Consulta')
    cx = encrypt(models.CharField(max_length=100, default=''))
    data = encrypt(models.DateField(default=timezone.now))
    alta = models.DecimalField(decimal_places=0, max_digits=3, default=0)
    pesoPreOp = models.FloatField(default=0)
    paciente = models.ForeignKey(Pacientes, on_delete=models.CASCADE)

    def getExcesso(self):
        return round(self.pesoPreOp - self.paciente.getPesoIdeal(), 2)

    def getIMCCir(self):
        return round(self.pesoPreOp/(self.paciente.altura*self.paciente.altura), 2)

    def getIMCCon(self):
        return round(self.peso/(self.paciente.altura*self.paciente.altura), 2)

    def getPerda(self, peso=None):
        cir = Consulta.objects.filter(paciente=self.paciente, evento="Cirurgia").last()
        if not peso:
            return round(cir.pesoPreOp - self.peso, 2)
        else:
            return round(cir.pesoPreOp - peso, 2)

    def getPerdaPerc(self, peso=None):
        cir = Consulta.objects.filter(paciente=self.paciente, evento="Cirurgia").last()
        if not peso:
            return round(100 * (self.getPerda()/cir.getExcesso()), 2)
        else:
            return round(100 * (self.getPerda(peso)/cir.getExcesso()), 2)
