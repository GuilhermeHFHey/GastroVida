from django.forms import ModelForm
from app.models import Pacientes

# Create the form class.


class PacientesForm(ModelForm):
    class Meta:
        model = Pacientes
        fields = ['nome', 'idade', 'peso',
                  'altura', 'pesoPreOperatorio', 'fumo']
