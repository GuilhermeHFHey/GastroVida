from django.forms import ModelForm
from app.models import Pacientes

# Create the form class.


class PacientesForm(ModelForm):
    class Meta:
        model = Pacientes
        fields = ['nome', 'idade', 'sexo', 'imc', 'altura',
                  'ca', 'rcq', 'gc', 'cx', 'data', 'alta',
                  'tpo', 'mes1', 'mes3', 'mes6', 'mes9', 'ano1',
                  'ano2', 'ano3', 'ano4', 'ano5']
