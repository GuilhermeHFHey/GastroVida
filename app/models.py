from django.db import models

# Create your models here.
class Pacientes(models.Model):
    nome = models.CharField(max_length=150)
    idade = models.IntegerField()
    peso = models.FloatField(max_length=30)