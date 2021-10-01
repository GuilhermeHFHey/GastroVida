from django.db import models
from django.forms.fields import ChoiceField

# Create your models here.


class Profissional(models.Model):
    nome = models.CharField(max_length=150)
    login = models.IntegerField()
    senha = models.IntegerField()
    prof = models.CharField(max_length=150)


class Pacientes(models.Model):
    SEXO = (
        ('M', 'Masculino'),
        ('F', 'Feminino'),
    )
    nome = models.CharField(max_length=150, blank=True, null=True)
    idade = models.IntegerField(blank=True, null=True)
    sexo = models.CharField(blank=True, null=True, max_length=1, choices=SEXO)
    imc = models.FloatField(blank=True, null=True, max_length=4)
    altura = models.FloatField(blank=True, null=True,
                               max_length=30)
    ca = models.IntegerField(blank=True, null=True)  # circunferencia abdominal
    rcq = models.IntegerField(blank=True, null=True)  # relacao cintura quadril
    gc = models.FloatField(blank=True, null=True, max_length=5)
    cx = models.CharField(blank=True, null=True, max_length=100)
    data = models.DateField(blank=True, null=True)
    alta = models.IntegerField(blank=True, null=True)
    tpo = models.IntegerField(blank=True, null=True)
    mes1 = models.FloatField(max_length=4, blank=True, null=True)
    mes3 = models.FloatField(max_length=4, blank=True, null=True)
    mes6 = models.FloatField(max_length=4, blank=True, null=True)
    mes9 = models.FloatField(max_length=4, blank=True, null=True)
    ano1 = models.FloatField(max_length=4, blank=True, null=True)
    ano2 = models.FloatField(max_length=4, blank=True, null=True)
    ano3 = models.FloatField(max_length=4, blank=True, null=True)
    ano4 = models.FloatField(max_length=4, blank=True, null=True)
    ano5 = models.FloatField(max_length=4, blank=True, null=True)
    proficional = models.ManyToManyField(Profissional)

    # peso = models.FloatField(max_length=30, blank=True)
    # pesoPreOperatorio = models.FloatField(
    #     max_length=30, blank=True)  # nao tem na lista do excel,
    # fumo = models.BooleanField(blank=True)  # nao tem na lista do excel,
    # perguntar se esta vai ser a relacao de tempo entre cada consulta mesmo
