from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from app.models import Pacientes
# Register your models here.


@admin.register(Pacientes)
class PacienteAdmin(ImportExportModelAdmin):
    list_display = ('nome', 'idade', 'sexo', 'imc', 'altura',
                    'ca', 'rcq', 'gc', 'cx', 'data', 'alta',
                    'tpo', 'mes1', 'mes3', 'mes6', 'mes9', 'ano1',
                    'ano2', 'ano3', 'ano4', 'ano5')
