from typing import List
from dateutil.relativedelta import relativedelta
from django import contrib
from django.forms.fields import NullBooleanField
from django.shortcuts import render, redirect
from scipy.sparse import data
from app.forms import LoginForm, PacientesForm, RegisterForm, ConsultaForm
from app.models import Pacientes, Profissional, Consulta
from django.template.loader import render_to_string
from django.http import JsonResponse
from django.contrib import messages
from tablib import Dataset
from django.core.paginator import Paginator
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import recall_score, roc_curve, accuracy_score, f1_score, precision_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from django.contrib.auth import authenticate, login, get_user_model, logout
from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import base64
import ast
import uuid
from datetime import date, datetime
from django.db.models import Max
# Create your views here.


def loginPage(request):
    if request.user.is_authenticated:
        logout(request)

    form = LoginForm(request.POST or None)
    data = {"form":form}
    if request.method == "POST":
        usuario = request.POST.get('usuario')
        senha = request.POST.get('senha')
        usuario = decrypt(usuario, key).decode("utf-8")
        senha = decrypt(senha, key).decode("utf-8")
        user = authenticate(request, username=usuario, password=senha)
        if user is not None:
            login(request, user)
            return JsonResponse(data={'message':'Login Valido'}, status=200)
        else:
            return JsonResponse(status=404)
    return render(request, 'login.html', data)


User = get_user_model()
def registerPage(request):
    form = RegisterForm(request.POST or None)
    data = {"form": form}
    if request.method == "POST" and form.is_valid():
        nome = form.cleaned_data.get("nome")
        username = form.cleaned_data["username"]
        password = form.cleaned_data["password"]
        prof = form.cleaned_data["prof"]
        newUser = User.objects.create_user(nome=nome, prof=prof, username=username, password=password)
    return render(request, "register.html", data)
 

def initial(request):
    if request.user.is_authenticated:
        return render(request, 'home.html')
    else:
        return redirect('login')


def home(request):
    if request.user.is_authenticated:
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
    else:
        return redirect('login')


def form(request):
    if request.user.is_authenticated:
        data = {}
        data['form'] = PacientesForm()
        return render(request, 'form.html', data)
    else:
        return redirect('login')


def create(request):
    if request.method == "POST":
        dados = request.POST.get('dados')
        dados = decrypt(dados, key).decode("utf-8")
        dados = ast.literal_eval(dados)
        nome = dados['nome']
        dataNasc = datetime.strptime(dados['dataNasc'], '%d/%m/%Y').date() if dados['dataNasc'] != '' else datetime.now()
        sexo = dados['sexo']
        altura = float(dados['altura']) if dados['altura'] != '' else 0
        cx = dados['cx']
        pesoPreOp = float(dados['pesoPreOp']) if dados['pesoPreOp'] != '' else 0
        data = datetime.strptime(dados['data'], '%d/%m/%Y').date() if dados['data'] != '' else datetime.now()
        alta = float(dados['alta']) if dados['altura'] != '' else 0
        proficional = (int(i) for i in dados['proficional'])
        pac = Pacientes(nome=nome, dataNasc=dataNasc, sexo=sexo, altura=altura, cx=cx,
         data=data, alta=alta, pesoPreOp=pesoPreOp)
        pac.save()
        for p in proficional:
            pac.proficional.add(p)
        return JsonResponse(data={'message':'Paciente registrado'}, status=200)


def view(request, pk):
    if request.user.is_authenticated:
        data = {}
        data['db'] = Pacientes.objects.get(pk=pk)
        data['con'] = Consulta.objects.filter(paciente=data['db'])
        return render(request, 'view.html', data)
    else:
        return redirect('login')


def edit(request, pk):
    if request.user.is_authenticated:
        data = {}
        data['pk'] = pk
        data['form'] = ConsultaForm()
        return render(request, 'edit.html', data)
    else:
        return redirect('login')


def update(request, pk):
    if request.method == "POST":
        dados = request.POST.get('dados')
        dados = decrypt(dados, key).decode("utf-8")
        dados = ast.literal_eval(dados)
        peso = float(dados['peso']) if dados['peso'] != '' else 0
        ca = float(dados['ca']) if dados['ca'] != '' else 0
        rcq = float(dados['rcq']) if dados['rcq'] != '' else 0
        gc = float(dados['gc']) if dados['gc'] != '' else 0
        data = datetime.strptime(dados['data'], '%d/%m/%Y').date() if dados['data'] != '' else datetime.now()
        pac = Pacientes.objects.get(pk=pk)
        numConsulta = ((data - pac.data).days) // 30
        lastCon = Consulta.objects.filter(paciente=pac, numConsulta=numConsulta)
        if not lastCon:
            con = Consulta(numConsulta=numConsulta, peso=peso, ca=ca, rcq=rcq, gc=gc, data=data, paciente=pac)
            con.save()
            return JsonResponse(data={'message':'Consulta registrada'}, status=200)
        else:
            return JsonResponse(data={'message':'Consulta já registrada nesse periodo'}, status=404)


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
            if paciente[0] == "CHEGA":
                break
            value = Pacientes(
                    nome="",
                    dataNasc=datetime.now() - relativedelta(years=paciente[1]),
                    sexo=paciente[2],
                    altura=float(paciente[4]),
                    cx=paciente[8],
                    data=paciente[9],
                    alta=float(paciente[10]) if paciente[10] is not None else 0,
                    pesoPreOp=float(paciente[0]),
            )
            p = Profissional.objects.get(pk=1)
            value.save()
            value.proficional.add(p)
            if paciente[12] != None:
                consulta = Consulta(
                        numConsulta=1,
                        peso=paciente[0]-(value.getExcesso()*paciente[12]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(months=1),
                        paciente=value
                )
                consulta.save()
            if paciente[13] != None:
                consulta = Consulta(
                        numConsulta=2,
                        peso=paciente[0]-(value.getExcesso()*paciente[13]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(months=3),
                        paciente=value
                )
                consulta.save()
            if paciente[14] != None:
                consulta = Consulta(
                        numConsulta=3,
                        peso=paciente[0]-(value.getExcesso()*paciente[14]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(months=6),
                        paciente=value
                )
                consulta.save()
            if paciente[15] != None:
                consulta = Consulta(
                        numConsulta=4,
                        peso=paciente[0]-(value.getExcesso()*paciente[15]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(months=9),
                        paciente=value
                )
                consulta.save()
            if paciente[16] != None:
                consulta = Consulta(
                        numConsulta=5,
                        peso=paciente[0]-(value.getExcesso()*paciente[16]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(years=1),
                        paciente=value
                )
                consulta.save()
            if paciente[17] != None:
                consulta = Consulta(
                        numConsulta=6,
                        peso=paciente[0]-(value.getExcesso()*paciente[17]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(years=2),
                        paciente=value
                )
                consulta.save()
            if paciente[18] != None:
                consulta = Consulta(
                        numConsulta=7,
                        peso=paciente[0]-(value.getExcesso()*paciente[18]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(years=3),
                        paciente=value
                )
                consulta.save()
            if paciente[19] != None:
                consulta = Consulta(
                        numConsulta=8,
                        peso=paciente[0]-(value.getExcesso()*paciente[19]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(years=4),
                        paciente=value
                )
                consulta.save()
            if paciente[20] != None:
                consulta = Consulta(
                        numConsulta=9,
                        peso=paciente[0]-(value.getExcesso()*paciente[20]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(years=5),
                        paciente=value
                )
                consulta.save()


    return render(request, 'import.html')

def getNumConsulta(data1, data2):
    numConsuta =(data1 - data2).days // 30
    if numConsuta == 1: return 1
    elif numConsuta == 3: return 2
    elif numConsuta == 6: return 3
    elif numConsuta == 9: return 4
    elif numConsuta == 12: return 5
    elif numConsuta == 24: return 6
    elif numConsuta == 36: return 7
    elif numConsuta == 48: return 8
    else: return 9


def dataFrame():
    d = {'pacienteId': Pacientes.objects.values_list("id", flat=True)}
    df = pd.DataFrame(data=d)
    if df is not None:
        abandonos = []
        ultimaCon = []
        Cons = np.zeros((len(df.index), 10))
        ConsMenosUlt = np.zeros((len(df.index), 9))
        for index, row in df.iterrows():
            pac = Pacientes.objects.get(pk=row["pacienteId"])
            consultas = Consulta.objects.filter(paciente=pac)
            Cons[index][0] = ((date.today() - pac.data).days)
            if not consultas:
                if getNumConsulta(date.today(), pac.data) > 2:
                    abandonos.append(True)
                else:
                    abandonos.append(False)
                ultimaCon.append(0)
                continue

            ultima = consultas.last()
            numConsultaHoje = getNumConsulta(date.today(), ultima.data)
            consultasMenosUlt = consultas.exclude(numConsulta=ultima.numConsulta)

            if (numConsultaHoje - ultima.numConsulta) > 2:
                abandonos.append(True)
            else:
                abandonos.append(False)

            for con in consultas:
                Cons[index][int(con.numConsulta)] = float(con.peso)

            for con in consultasMenosUlt:
                ConsMenosUlt[index][int(con.numConsulta)-1] = con.peso

            ultimaCon.append(ultima.peso)

        df["abandono"] = abandonos
        df["consultas"] = Cons.tolist()
        df["consultasMenosultima"] = ConsMenosUlt.tolist()
        df["ultimaCon"] = ultimaCon

        return df
    else:
        return None

df = dataFrame()

def ClassificadorLOO():
    global df
    X = np.stack(df["consultas"], axis=0)
    y = df["abandono"]
    # y_true, y_pred, probs = [], [], []

    # cv = LeaveOneOut()
    # for train_index, test_index in cv.split(X):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     rf = RandomForestClassifier()
    #     rf.fit(X_train, y_train)
    #     y_true.append(y_test)
    #     y_pred.append(rf.predict(X_test)[0])
    #     probs.append(rf.predict_proba(X_test)[:, 1])

    # y_pred = np.array(y_pred)
    # y_true = np.array(y_true)
    # probs = np.array(probs)
    # acc = accuracy_score(y_true, y_pred)
    # pre = precision_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred)
    # auc = roc_auc_score(y_true, probs)
    # print("="*30)
    # print("Testes com base original")
    # print("Acuracia: ", acc)
    # print("Area ROC: ", auc)
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # print("Sencibilidade :", tp / (tp+fn))
    # print("Especificidade: ", tn / (tn+fp))
    # plot_matrix(confusion_matrix(y_true, y_pred))
    # print("="*30)

    # fper, tper, thresholds = roc_curve(y, probs)
    # plot_roc_curve(fper, tper)

    rus = RandomUnderSampler()
    X, y = rus.fit_resample(X, y)
    # y_true, y_pred, probs = [], [], []

    # cv = LeaveOneOut()
    # for train_index, test_index in cv.split(X):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     rf = RandomForestClassifier()
    #     rf.fit(X_train, y_train)
    #     y_true.append(y_test)
    #     y_pred.append(rf.predict(X_test)[0])
    #     probs.append(rf.predict_proba(X_test)[:, 1])

    # acc = accuracy_score(y_true, y_pred)
    # pre = precision_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred)
    # auc = roc_auc_score(y_true, probs)
    # print("="*30)
    # print("Testes com base balanceada")
    # print("Acuracia: ", acc)
    # print("Area ROC: ", auc)
    # print("Sencibilidade :", recall_score(y_true, y_pred))
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # print("Especificidade: ", tn / (tn+fp))
    # plot_matrix(confusion_matrix(y_true, y_pred))
    # print("="*30)

    # fper, tper, thresholds = roc_curve(y, probs)
    # plot_roc_curve(fper, tper)
    rf = RandomForestClassifier()
    rf.fit(X, y)
    # ,acc, pre, f1, auc
    return rf

"""
def ClassificadorKF():
    global df
    print("INICIO KF")
    X = df[["tpo", "mes1", "mes3", "mes6", "mes9",
            "ano1", "ano2", "ano3", "ano4", "ano5"]]
    y = df["Abandono"]
    y_true, y_pred, probs = [], [], []

    cv = KFold(n_splits=5)
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_true = y_true + y_test.tolist()
        y_pred = y_pred + rf.predict(X_test).tolist()
        probTrue = []
        for trues in rf.predict_proba(X_test):
            array = [trues[1]]
            probTrue = probTrue + array
        probs = probs + probTrue

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)
    print("="*30)
    print("Testes com base original KF")
    print("Acuracia: ", acc)
    print("Area ROC: ", auc)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("Sencibilidade :", tp / (tp+fn))
    print("Especificidade: ", tn / (tn+fp))
    plot_matrix(confusion_matrix(y_true, y_pred))
    print("="*30)

    fper, tper, thresholds = roc_curve(y, probs)
    plot_roc_curve(fper, tper)

    rus = RandomUnderSampler()
    X, y = rus.fit_resample(X, y)
    y_true, y_pred, probs = [], [], []

    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_true = y_true + y_test.tolist()
        y_pred = y_pred + rf.predict(X_test).tolist()
        probTrue = []
        for trues in rf.predict_proba(X_test):
            array = [trues[1]]
            probTrue = probTrue + array
        probs = probs + probTrue

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)
    print("="*30)
    print("Testes com base balanceada KF")
    print("Acuracia: ", acc)
    print("Area ROC: ", auc)
    print("Sencibilidade :", recall_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("Especificidade: ", tn / (tn+fp))
    plot_matrix(confusion_matrix(y_true, y_pred))
    print("="*30)

    fper, tper, thresholds = roc_curve(y, probs)
    plot_roc_curve(fper, tper)


def plot_matrix(matrix):
    ax = plt.subplot()
    sn.heatmap(matrix, annot=True, fmt="d")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.savefig("ConfusionMatrix.png")


def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig("ROC.png")
"""

def Regressor():
    global df
    X = np.stack(df["consultasMenosultima"], axis=0)
    y = df["ultimaCon"]
    # y_true, y_pred = [], []

    # cv = LeaveOneOut()
    # for train_index, test_index in cv.split(X):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     lr = LinearRegression()
    #     lr.fit(X_train, y_train)
    #     y_true.append(y_test)
    #     y_pred.append(lr.predict(X_test)[0])

    # mae = mean_absolute_error(y_true, y_pred)
    # mse = mean_squared_error(y_true, y_pred, squared=True)
    # rmse = mean_squared_error(y_true, y_pred, squared=False)
    # print("="*30)
    # print("LOO")
    # print(mae)
    # print(mse)
    # print(rmse)
    # print("="*30)

    # cv = KFold(n_splits=5)
    # for train_index, test_index in cv.split(X):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     lr = LinearRegression()
    #     lr.fit(X_train, y_train)
    #     y_true = y_true + y_test.tolist()
    #     y_pred = y_pred + lr.predict(X_test).tolist()

    # r2 = r2_score(y_true, y_pred)
    # mae = mean_absolute_error(y_true, y_pred)
    # mse = mean_squared_error(y_true, y_pred, squared=True)
    # rmse = mean_squared_error(y_true, y_pred, squared=False)
    # print("="*30)
    # print("KFOLD")
    # print(mae)
    # print(mse)
    # print(rmse)
    # print("="*30)

    lr = LinearRegression()
    lr.fit(X, y)
    # , mae, mse, rmse
    return lr

rf = ClassificadorLOO()
print("Classificador Pronto")

lr = Regressor()
print("Regressor Pronto")
# acc = round(acc, 2)*100
# pre = round(acc, 2)*100
# f1 = round(acc, 2)*100
# auc = round(acc, 2)*100


def PrevPred(request, pk):
    global rf, lr
    pac = Pacientes.objects.get(pk=pk)
    cons = Consulta.objects.filter(paciente=pac)
    consultas = np.zeros(10)
    consultasSemData = np.zeros(9)
    perdaPerc = np.zeros(9)
    consultas[0] = ((date.today() - pac.data).days)
    ult = 0
    for c in cons:
        consultas[int(c.numConsulta)] = c.peso
        consultasSemData[int(c.numConsulta)-1] = c.peso
        perdaPerc[int(c.numConsulta)-1] = c.getPerdaPerc()
        ult = int(c.numConsulta)
    
    prob = rf.predict_proba(consultas[np.newaxis, :])
    pdp = lr.predict(consultasSemData[np.newaxis, :])[0]
    perdaPerc[ult:] = pdp
    lines = pd.DataFrame({
        'Média':[25, 45, 64, 75, 80, 84, 84, 76, 0],
        'Paciente Previsto':perdaPerc,
    }, index=['Mês 1', 'Mês 3', 'Mês 6', 'Mês 9', 'Ano 1', 'Ano 2', 'Ano 3', 'Ano 4', 'Ano 5'])
    plt.figure()
    lines.plot.line()
    plt.savefig("./static/images/grafico.png")

    data = {}
    data['db'] = {}
    data['db'] = ({'prob': round(prob[0][1], 2)*100, 'pdp4': round(100 * (pdp/pac.getExcesso()), 2), 'id': pk})
    return render(request, 'prevPred.html', data)

"""
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
    data['db'] = ({'prob': round(prob[0][1], 2)*100, 'id': pk})
    return render(request, 'prediçao.html', data)


def Previsao(request, pk):
    global df, lr, mae, mse, rmse
    df = pd.DataFrame(list(Pacientes.objects.all().values()))
    paciente = df.loc[df["id"] == pk]
    pdp1 = 0.25
    pdp2 = 0.45
    pdp3 = 0.64
    try:
        pdp1 = paciente["mes1"].values[0]
    except Exception:
        pass
    try:
        pdp2 = paciente["mes3"].values[0]
    except Exception:
        pass
    try:
        pdp3 = paciente["mes6"].values[0]
    except Exception:
        pass
    paciente["PDP1"] = pdp1
    paciente["PDP2"] = pdp2
    paciente["PDP3"] = pdp3
    paciente = paciente[["PDP1", "PDP2", "PDP3"]]
    pdp4 = lr.predict(paciente)[0]
    lines = pd.DataFrame({
        'Média':[0.25, 0.45, 0.64, 0.758473684],
        'Paciente':[pdp1, pdp2, pdp3, pdp4]
    }, index=['Mês1', 'Mês3', 'Mês6', 'Mês9'])
    plt.figure()
    lines.plot.line()
    plt.savefig("grafico.png")
    data = {}
    data['db'] = {}
    data['db'] = ({'pdp4': round(pdp4, 2)*100, 'rmse': rmse,
                   'mae': mae, 'mse': mse, 'id': pk})

    return render(request, 'previsao.html', data)
"""

BLOCK_SIZE = 16
key = b"1234567890123456"

def pad(data):
    length = BLOCK_SIZE - (len(data) % BLOCK_SIZE)
    return data + chr(length)*length


def encrypt(message, key):
    IV = Random.new().read(BLOCK_SIZE)
    aes = AES.new(key, AES.MODE_CBC, IV)
    return base64.b64encode(IV + aes.encrypt(pad(message)))

def decrypt(encrypted, key):
    encrypted = base64.b64decode(encrypted)
    IV = encrypted[:BLOCK_SIZE]
    aes = AES.new(key, AES.MODE_CBC, IV)
    return unpad(aes.decrypt(encrypted[BLOCK_SIZE:]), BLOCK_SIZE)