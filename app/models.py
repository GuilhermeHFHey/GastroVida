from django.db import models
from django.forms.fields import ChoiceField
from django.contrib.auth.models import AbstractUser
from django_cryptography.fields import encrypt


# # Create your models here.


class Profissional(AbstractUser):
    nome = encrypt(models.CharField(max_length=150, editable=False))
    prof = encrypt(models.CharField(max_length=150, editable=False))

class Pacientes(models.Model):
    SEXO = (
        ('M', 'Masculino'),
        ('F', 'Feminino'),
    )
    nome = encrypt(models.CharField(max_length=150, blank=True, null=True))
    idade = encrypt(models.DecimalField(blank=True, null=True, decimal_places=0, max_digits=3))
    sexo = encrypt(models.CharField(blank=True, null=True, max_length=1, choices=SEXO))
    imc = encrypt(models.FloatField(blank=True, null=True, max_length=4))
    altura = encrypt(models.FloatField(blank=True, null=True,
                               max_length=30))
    ca = encrypt(models.DecimalField(blank=True, null=True, decimal_places=0, max_digits=3))  # circunferencia abdominal
    rcq = encrypt(models.DecimalField(blank=True, null=True, decimal_places=0, max_digits=3))  # relacao cintura quadril
    gc = encrypt(models.FloatField(blank=True, null=True, max_length=5))
    cx = encrypt(models.CharField(blank=True, null=True, max_length=100))
    data = encrypt(models.DateField(blank=True, null=True))
    alta = encrypt(models.DecimalField(blank=True, null=True, decimal_places=0, max_digits=3))
    tpo = encrypt(models.DecimalField(blank=True, null=True, decimal_places=0, max_digits=3))
    mes1 = encrypt(models.FloatField(max_length=4, blank=True, null=True))
    mes3 = encrypt(models.FloatField(max_length=4, blank=True, null=True))
    mes6 = encrypt(models.FloatField(max_length=4, blank=True, null=True))
    mes9 = encrypt(models.FloatField(max_length=4, blank=True, null=True))
    ano1 = encrypt(models.FloatField(max_length=4, blank=True, null=True))
    ano2 = encrypt(models.FloatField(max_length=4, blank=True, null=True))
    ano3 = encrypt(models.FloatField(max_length=4, blank=True, null=True))
    ano4 = encrypt(models.FloatField(max_length=4, blank=True, null=True))
    ano5 = encrypt(models.FloatField(max_length=4, blank=True, null=True))
    proficional = models.ManyToManyField(Profissional)

    # peso = models.FloatField(max_length=30, blank=True)
    # pesoPreOperatorio = models.FloatField(
    #     max_length=30, blank=True)  # nao tem na lista do excel,
    # fumo = models.BooleanField(blank=True)  # nao tem na lista do excel,
    # perguntar se esta vai ser a relacao de tempo entre cada consulta mesmo
