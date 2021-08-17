from django.db import models

# Create your models here.


class Pacientes(models.Model):
    nome = models.CharField(max_length=150, blank=True)
    idade = models.IntegerField(blank=True)
    # nao tem na lista do excel, perguntar ao DrRodrigo
    peso = models.FloatField(max_length=30, blank=True)
    altura = models.FloatField(max_length=30, blank=True)
    pesoPreOperatorio = models.FloatField(
        max_length=30, blank=True)  # nao tem na lista do excel,
    fumo = models.BooleanField(blank=True)  # nao tem na lista do excel,
    sexo = models.CharField(max_length=1)
    # perguntar se é atual ou pre operatorio
    imc = models.FloatField(max_length=4, blank=True)
    ca = models.IntegerField(blank=True)  # circunferencia abdominal
    rcq = models.IntegerField(blank=True)  # relacao cintura quadril
    # gc = no excel tem uma coluna "%GC", verificar com DrRodrigo
    # verificar se é possivel um dropdown
    # (lista com todos os tipos de cirurgia)
    cx = models.CharField(max_length=100)
    # verificar o que significa esta data
    # (primeira cirurgia? primeira consulta?)
    data = models.DateField()
    # alta = verificar o que significa e qual valor vai na tabela,
    #  data? alta do paciente?
    # tempo pos operatorio, ver o que significa este valor
    tpo = models.IntegerField(blank=True)
    # perguntar se esta vai ser a relacao de tempo entre cada consulta mesmo
    mes1 = models.FloatField(max_length=4, blank=True)
    mes3 = models.FloatField(max_length=4, blank=True)
    mes6 = models.FloatField(max_length=4, blank=True)
    mes9 = models.FloatField(max_length=4, blank=True)
    ano1 = models.FloatField(max_length=4, blank=True)
    ano2 = models.FloatField(max_length=4, blank=True)
    ano3 = models.FloatField(max_length=4, blank=True)
    ano4 = models.FloatField(max_length=4, blank=True)
    ano5 = models.FloatField(max_length=4, blank=True)
