# Generated by Django 3.2.5 on 2021-08-27 22:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_auto_20210827_1714'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pacientes',
            name='idade',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='pacientes',
            name='nome',
            field=models.CharField(blank=True, max_length=150, null=True),
        ),
    ]
