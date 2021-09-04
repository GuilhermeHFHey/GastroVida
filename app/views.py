from django.shortcuts import render, redirect
import numpy as np
from app.forms import PacientesForm
from app.models import Pacientes
from django.template.loader import render_to_string
from django.http import JsonResponse
from django.contrib import messages
from tablib import Dataset
from django.core.paginator import Paginator
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
# Create your views here.


def home(request):
    data = {}
    search = request.GET.get("q")
    if search:
        pacientes = Pacientes.objects.filter(nome__icontains=search)
    else:
        pacientes = Pacientes.objects.all()
    paginator = Paginator(pacientes, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    data['db'] = page_obj
    if request.is_ajax():
        html = render_to_string(
            template_name="index_partial.html",
            context={'db': pacientes}
        )
        data_dict = {"html_from_view": html}
        return JsonResponse(data=data_dict, safe=False)

    return render(request, 'index.html', data)


def form(request):
    data = {}
    data['form'] = PacientesForm()
    return render(request, 'form.html', data)


def create(request):
    form = PacientesForm(request.POST or None)
    if form.is_valid():
        form.save()
        return redirect('home')


def view(request, pk):
    data = {}
    data['db'] = Pacientes.objects.get(pk=pk)
    return render(request, 'view.html', data)


def edit(request, pk):
    data = {}
    data['db'] = Pacientes.objects.get(pk=pk)
    data['form'] = PacientesForm(instance=data['db'])
    return render(request, 'form.html', data)


def update(request, pk):
    data = {}
    data['db'] = Pacientes.objects.get(pk=pk)
    form = PacientesForm(request.POST or None, instance=data['db'])
    if form.is_valid():
        form.save()
        return redirect('home')


def delete(request, pk):
    db = Pacientes.objects.get(pk=pk)
    db.delete()
    return redirect('home')


def uploadExel(request):
    if request.method == 'POST':
        dataset = Dataset()
        dados = request.FILES['myfile']
        if not dados.name.endswith('xlsx'):
            messages.info(request, 'Formato Incorreto')
            return render(request, 'import.html')

        dadosImportados = dataset.load(dados.read(), format='xlsx')
        for paciente in dadosImportados:
            value = Pacientes(
                paciente[0],
                paciente[0],
                paciente[1],
                paciente[2],
                paciente[3],
                paciente[4],
                paciente[5],
                paciente[6],
                paciente[7],
                paciente[8],
                paciente[9],
                paciente[10],
                paciente[11],
                paciente[12],
                paciente[13],
                paciente[14],
                paciente[15],
                paciente[16],
                paciente[17],
                paciente[18],
                paciente[19],
                paciente[20]
            )
            value.save()
    return render(request, 'import.html')


def dataFrame():
    df = pd.DataFrame(list(Pacientes.objects.all().values()))
    df.drop(index=df.index[0], axis=0, inplace=True)
    df["mes1"].fillna(-1, inplace=True)
    df["mes3"].fillna(-1, inplace=True)
    df["mes6"].fillna(-1, inplace=True)
    df["mes9"].fillna(-1, inplace=True)
    df["ano1"].fillna(-1, inplace=True)
    df["ano2"].fillna(-1, inplace=True)
    df["ano3"].fillna(-1, inplace=True)
    df["ano4"].fillna(-1, inplace=True)
    df["ano5"].fillna(-1, inplace=True)
    Abandonos = []
    ProximasConsultas = []
    for index, row in df.iterrows():
        abandono = False
        proximaConsulta = ""
        consultaAtual = row["tpo"]/365
        if consultaAtual <= 0.07:
            proximaConsulta = "mes1"
        elif 0.07 < consultaAtual <= 0.25:
            proximaConsulta = "mes3"
        elif 0.25 < consultaAtual <= 0.5:
            proximaConsulta = "mes6"
            if row["mes3"] == -1 and row["mes1"] == -1:
                abandono = True
        elif 0.5 < consultaAtual <= 0.75:
            proximaConsulta = "mes9"
            if row["mes6"] == -1 and row["mes3"] == -1:
                abandono = True
        elif 0.75 < consultaAtual <= 1:
            proximaConsulta = "ano1"
            if row["mes9"] == -1 and row["mes6"] == -1:
                abandono = True
        elif 1 < consultaAtual <= 2:
            proximaConsulta = "ano2"
            if row["ano1"] == -1 and row["mes9"] == -1:
                abandono = True
        elif 2 < consultaAtual <= 3:
            proximaConsulta = "ano3"
            if row["ano1"] == -1 and row["ano2"] == -1:
                abandono = True
        elif 3 < consultaAtual <= 4:
            proximaConsulta = "ano4"
            if row["ano3"] == -1 and row["ano2"] == -1:
                abandono = True
        elif 4 < consultaAtual <= 5:
            proximaConsulta = "ano5"
            if row["ano4"] == -1 and row["ano3"] == -1:
                abandono = True
        else:
            if row["ano5"] == -1 and row["ano4"] == -1:
                abandono = True

        Abandonos.append(abandono)
        ProximasConsultas.append(proximaConsulta)

    df["Abandono"] = Abandonos
    df["ProximaConsulta"] = ProximasConsultas

    return df


# def Previsão(pk):
#     global df
#     paciente = df.loc[df["id"] == pk]
#     X = df[["mes1", "mes3", "mes6", "mes9", "ano1", "ano2", "ano3", "ano4"]]
#     y = df["ano5"]
#     lr = LinearRegression()
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42)
#     lr.fit(X_train, y_train)
#     respostas = lr.predict(X_test)
#     r2 = lr.score(respostas, y_test)

def Classificador():
    global df
    X = df[["tpo", "mes1", "mes3", "mes6", "mes9",
            "ano1", "ano2", "ano3", "ano4", "ano5"]]
    y = df["Abandono"]
    rf = RandomForestClassifier()
    rf.fit(X, y)
    cv = LeaveOneOut()
    scores = cross_val_score(rf, X, y, cv=cv, n_jobs=1)
    acc = np.mean(np.absolute(scores))
    return rf, acc


df = dataFrame()
rf, acc = Classificador()


def Prediçao(request, pk):
    global df, rf, acc
    paciente = df.loc[df["id"] == pk]
    paciente = paciente[["tpo", "mes1", "mes3", "mes6", "mes9",
                         "ano1", "ano2", "ano3", "ano4", "ano5"]]
    prob = rf.predict_proba(paciente)
    data = {}
    data['db'] = {}
    data['db'] = ({'prob': prob[0][1], 'acc': acc, 'id': pk})
    return render(request, 'previsoes.html', data)
