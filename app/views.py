from django.shortcuts import render, redirect
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
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, accuracy_score, f1_score, precision_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
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
    pdps1 = []
    pdps2 = []
    pdps3 = []
    pdps4 = []
    for index, row in df.iterrows():
        pdp1 = 0.25
        pdp2 = 0.45
        pdp3 = 0.64
        pdp4 = 0.75
        if row["mes1"] != -1:
            pdp1 = row["mes1"]
        if row["mes3"] != -1:
            pdp2 = row["mes3"]
        if row["mes6"] != -1:
            pdp3 = row["mes6"]
        if row["mes9"] != -1:
            pdp4 = row["mes9"]

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
        pdps1.append(pdp1)
        pdps2.append(pdp2)
        pdps3.append(pdp3)
        pdps4.append(pdp4)
        Abandonos.append(abandono)
        ProximasConsultas.append(proximaConsulta)

    df["Abandono"] = Abandonos
    df["ProximaConsulta"] = ProximasConsultas
    df["PDP1"] = pdps1
    df["PDP2"] = pdps2
    df["PDP3"] = pdps3
    df["PDP4"] = pdps4

    return df


df = dataFrame()


def Classificador():
    global df
    X = df[["tpo", "mes1", "mes3", "mes6", "mes9",
            "ano1", "ano2", "ano3", "ano4", "ano5"]]
    y = df["Abandono"]
    rus = RandomUnderSampler()
    y_true, y_pred, probs = [], [], []

    cv = LeaveOneOut()
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, y_train = rus.fit_resample(X_train, y_train)
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_true.append(y_test)
        y_pred.append(rf.predict(X_test)[0])
        probs.append(rf.predict_proba(X_test)[:, 1])

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)

    # prob = rf.predict_proba(X)
    # prob = prob[:, 1]
    # fper, tper, thresholds = roc_curve(y, prob)
    # plot_roc_curve(fper, tper)

    rf = RandomForestClassifier()
    rf.fit(X, y)
    return rf, acc, pre, f1, auc


# def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()


def Regressor():
    global df
    X = df[["PDP1", "PDP2", "PDP3"]]
    y = df["PDP4"]
    y_true, y_pred = [], []

    cv = LeaveOneOut()
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_true.append(y_test)
        y_pred.append(lr.predict(X_test)[0])

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    lr = LinearRegression()
    lr.fit(X, y)

    return lr, r2, mae, mse


rf, acc, pre, f1, auc = Classificador()
print("Classificador Pronto")
lr, r2, mae, mse = Regressor()
print("Regressor Pronto")
acc = round(acc, 2)*100
pre = round(acc, 2)*100
f1 = round(acc, 2)*100
auc = round(acc, 2)*100
r2 = round(r2, 2)*100


def Prediçao(request, pk):
    global rf, acc, pre, f1, auc
    df = pd.DataFrame(list(Pacientes.objects.all().values()))
    df["mes1"].fillna(-1, inplace=True)
    df["mes3"].fillna(-1, inplace=True)
    df["mes6"].fillna(-1, inplace=True)
    df["mes9"].fillna(-1, inplace=True)
    df["ano1"].fillna(-1, inplace=True)
    df["ano2"].fillna(-1, inplace=True)
    df["ano3"].fillna(-1, inplace=True)
    df["ano4"].fillna(-1, inplace=True)
    df["ano5"].fillna(-1, inplace=True)
    paciente = df.loc[df["id"] == pk]
    paciente = paciente[["tpo", "mes1", "mes3", "mes6", "mes9",
                         "ano1", "ano2", "ano3", "ano4", "ano5"]]
    prob = rf.predict_proba(paciente)
    data = {}
    data['db'] = {}
    data['db'] = ({'prob': round(prob[0][1], 2)*100, 'acc': acc,
                  'pre': pre, 'f1': f1, 'auc': auc, 'id': pk})
    return render(request, 'prediçao.html', data)


def Previsao(request, pk):
    global df, lr, r2, mae, mse
    df = pd.DataFrame(list(Pacientes.objects.all().values()))
    paciente = df.loc[df["id"] == pk]
    pdp1 = 0.25
    pdp2 = 0.45
    pdp3 = 0.64
    try:
        pdp1 = paciente["mes1"]
    except Exception:
        pass
    try:
        pdp2 = paciente["mes3"]
    except Exception:
        pass
    try:
        pdp3 = paciente["mes6"]
    except Exception:
        pass
    paciente["PDP1"] = pdp1
    paciente["PDP2"] = pdp2
    paciente["PDP3"] = pdp3
    paciente = paciente[["PDP1", "PDP2", "PDP3"]]
    print(paciente)
    pdp4 = lr.predict(paciente)
    data = {}
    data['db'] = {}
    data['db'] = ({'pdp4': round(pdp4[0], 2)*100, 'r2': r2,
                  'mae': mae, 'mse': mse, 'id': pk})
    return render(request, 'previsao.html', data)
