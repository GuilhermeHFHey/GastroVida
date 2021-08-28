from import_export import resources
from app.models import Pacientes


class PacienteResource(resources.ModelResource):
    class meta:
        model = Pacientes
