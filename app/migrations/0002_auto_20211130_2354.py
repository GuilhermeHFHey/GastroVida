# Generated by Django 3.2.7 on 2021-12-01 02:54

from django.db import migrations, models
import django_cryptography.fields


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='consulta',
            name='ca',
            field=django_cryptography.fields.encrypt(models.DecimalField(decimal_places=0, default=0, max_digits=3)),
        ),
        migrations.AlterField(
            model_name='consulta',
            name='evento',
            field=django_cryptography.fields.encrypt(models.CharField(choices=[('Consulta', 'Consulta'), ('Cirurgia', 'Cirurgia')], default='Consulta', max_length=8)),
        ),
        migrations.AlterField(
            model_name='consulta',
            name='gc',
            field=django_cryptography.fields.encrypt(models.FloatField(default=0, max_length=5)),
        ),
        migrations.AlterField(
            model_name='consulta',
            name='numConsulta',
            field=models.FloatField(default=0),
        ),
        migrations.AlterField(
            model_name='consulta',
            name='peso',
            field=models.FloatField(default=0, max_length=5),
        ),
        migrations.AlterField(
            model_name='consulta',
            name='rcq',
            field=django_cryptography.fields.encrypt(models.DecimalField(decimal_places=0, default=0, max_digits=3)),
        ),
    ]