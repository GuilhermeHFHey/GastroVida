from django.contrib import admin
from django.contrib.auth import admin as authAdmin
from import_export.admin import ImportExportModelAdmin
from app.models import Pacientes, Profissional
# Register your models here.

admin.site.register(Profissional, authAdmin.UserAdmin)

@admin.register(Pacientes)
class PacienteAdmin(ImportExportModelAdmin):
    list_display = ('nome', 'dataNasc', 'sexo', 'altura')
