from typing import List
from dateutil.relativedelta import relativedelta
from django import contrib
from django.forms.fields import NullBooleanField
from django.shortcuts import render, redirect
from joblib.logger import PrintTime
from scipy.sparse import data
from scipy.sparse.construct import random
from app.forms import LoginForm, PacientesForm, RegisterForm, ConsultaForm
from app.models import Pacientes, Profissional, Consulta
from django.template.loader import render_to_string
from django.http import JsonResponse
from django.contrib import messages
from tablib import Dataset
from django.core.paginator import Paginator
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
from datetime import date, datetime, timedelta
from django.db.models import Max
import random
# Create your views here.


def loginPage(request):
    if request.user.is_authenticated:
        logout(request)

    form = LoginForm(request.POST or None)
    data = {"form":form}
    if request.method == "POST":
        dados = request.POST.get('dados')
        dados = decrypt(dados, key).decode("utf-8")
        dados = ast.literal_eval(dados)
        print(dados)
        usuario = dados['usuario']
        senha = dados['senha']
        user = authenticate(request, username=usuario, password=senha)
        if user is not None:
            login(request, user)
            return JsonResponse(data={'message':'Login Valido'}, status=200)
        else:
            return JsonResponse(status=404)
    return render(request, 'login.html', data)


User = get_user_model()
def registerPage(request):
    if request.user.is_authenticated:
        if request.user.is_superuser:
            data = {}
            data['form'] = RegisterForm()
            if request.method == "POST":
                dados = request.POST.get('dados')
                dados = decrypt(dados, key).decode("utf-8")
                dados = ast.literal_eval(dados)
                nome = dados['nome']
                username = dados['username']
                password = dados['password']
                prof = dados['profissao']
                newUser = User.objects.create_user(nome=nome, prof=prof, username=username, password=password)
                return JsonResponse(data={'message':'Profissional Registrado'}, status=200)
            return render(request, "register.html", data)
        else:
            return redirect('initial')
    else:
        return redirect('login') 


def initial(request):
    if request.user.is_authenticated:
        return render(request, 'home.html')
    else:
        return redirect('login')


def home(request):
    if request.user.is_authenticated:
        p = Profissional.objects.get(username=request.user)
        data = {}
        search = request.GET.get("q")
        if request.user.is_superuser:
            if search:
                pacientes = Pacientes.objects.filter(nome__icontains=search)
            else:
                pacientes = Pacientes.objects.all()
        else:
            if search:
                pacientes = Pacientes.objects.filter(nome__icontains=search).filter(proficional=p)
            else:
                pacientes = Pacientes.objects.filter(proficional=p)
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
        proficional = (int(i) for i in dados['proficional'])
        pac = Pacientes(nome=nome, dataNasc=dataNasc, sexo=sexo, altura=altura)
        pac.save()
        for p in proficional:
            pac.proficional.add(p)
        return JsonResponse(data={'message':'Paciente registrado'}, status=200)


def cirurgiaForm(request, pk):
    if request.user.is_authenticated:
        data = {}
        data['pac'] = Pacientes.objects.get(pk=pk)
        data['form'] = ConsultaForm(initial={'data': '', 'alta': '', 'pesoPreOp':''})
        return render(request, 'cirurgia.html', data)
    else:
        return redirect('login')


def createCirurgia(request, pk):
    if request.method == "POST":
        dados = request.POST.get('dados')
        dados = decrypt(dados, key).decode("utf-8")
        dados = ast.literal_eval(dados)
        cx = dados['cx']
        data = datetime.strptime(dados['data'], '%d/%m/%Y').date() if dados['data'] != '' else datetime.now()
        alta = float(dados['alta']) if dados['alta'] != '' else 0
        pesoPreOp = float(dados['pesoPreOp']) if dados['pesoPreOp'] != '' else 0
        pac = Pacientes.objects.get(pk=pk)
        cirurgia = Consulta(evento="Cirurgia", cx=cx, data=data, alta=alta, pesoPreOp=pesoPreOp, paciente=pac)
        cirurgia.save()
        return JsonResponse(data={'message':'Paciente registrado'}, status=200)


def editCir(request, pk):
    if request.user.is_authenticated:
        data = {}
        data['con'] = Consulta.objects.get(pk=pk)
        data['pac'] = data['con'].paciente
        data['form'] = ConsultaForm(instance=data['db'])
        return render(request, 'cirurgia.html', data)
    else:
        return redirect('login')


def updateCir(request, pk):
    if request.method == "POST":
        dados = request.POST.get('dados')
        dados = decrypt(dados, key).decode("utf-8")
        dados = ast.literal_eval(dados)
        cx = dados['cx']
        data = datetime.strptime(dados['data'], '%d/%m/%Y').date() if dados['data'] != '' else datetime.now()
        alta = float(dados['alta']) if dados['alta'] != '' else 0
        pesoPreOp = float(dados['pesoPreOp']) if dados['pesoPreOp'] != '' else 0
        con = Consulta.objects.get(pk=pk)
        con.cx=cx
        con.alta=alta
        con.pesoPreOp=pesoPreOp
        con.data=data
        con.save()
        return JsonResponse(data={'message':'Consulta registrada'}, status=200)


def editPac(request, pk):
    if request.user.is_authenticated:
        data = {}
        data['db'] = Pacientes.objects.get(pk=pk)
        data['form'] = PacientesForm(instance=data['db'])
        return render(request, 'form.html', data)
    else:
        return redirect('login')


def updatePac(request, pk):
    if request.method == "POST":
        dados = request.POST.get('dados')
        dados = decrypt(dados, key).decode("utf-8")
        dados = ast.literal_eval(dados)
        nome = dados['nome']
        dataNasc = datetime.strptime(dados['dataNasc'], '%d/%m/%Y').date() if dados['dataNasc'] != '' else datetime.now()
        sexo = dados['sexo']
        altura = float(dados['altura']) if dados['altura'] != '' else 0
        proficional = (int(i) for i in dados['proficional'])
        pac = Pacientes.objects.get(pk=pk)
        pac.nome=nome
        pac.dataNasc=dataNasc
        pac.sexo=sexo
        pac.altura=altura
        pac.save()
        for p in proficional:
            pac.proficional.add(p)
        return JsonResponse(data={'message':'Paciente registrado'}, status=200)


def editCon(request, pk):
    if request.user.is_authenticated:
        data = {}
        data['db'] = Consulta.objects.get(pk=pk)
        data['pac'] = data['db'].paciente
        data['form'] = ConsultaForm(instance=data['db'])
        return render(request, 'edit.html', data)
    else:
        return redirect('login')


def updateCon(request, pk):
    if request.method == "POST":
        dados = request.POST.get('dados')
        dados = decrypt(dados, key).decode("utf-8")
        dados = ast.literal_eval(dados)
        peso = float(dados['peso']) if dados['peso'] != '' else 0
        ca = float(dados['ca']) if dados['ca'] != '' else 0
        rcq = float(dados['rcq']) if dados['rcq'] != '' else 0
        gc = float(dados['gc']) if dados['gc'] != '' else 0
        data = datetime.strptime(dados['data'], '%d/%m/%Y').date() if dados['data'] != '' else datetime.now()
        con = Consulta.objects.get(pk=pk)
        cir = Consulta.objects.filter(paciente=con.paciente, evento="Cirurgia").last()
        if cir:
            numConsulta = getNumConsulta(data, cir.data)
        else:
            numConsulta = 0
        con.numConsulta=numConsulta
        con.peso=peso
        con.ca=ca
        con.rcq=rcq
        con.gc=gc
        con.data=data
        con.gc=gc
        con.save()
        return JsonResponse(data={'message':'Consulta registrada'}, status=200)


def view(request, pk):
    if request.user.is_authenticated:
        data = {}
        data['db'] = Pacientes.objects.get(pk=pk)
        data['con'] = Consulta.objects.filter(paciente=data['db'], evento="Consulta").order_by("numConsulta", "data")
        data['cir'] = Consulta.objects.filter(paciente=data['db'], evento="Cirurgia")
        return render(request, 'view.html', data)
    else:
        return redirect('login')


def edit(request, pk):
    if request.user.is_authenticated:
        data = {}
        data['pac'] = Pacientes.objects.get(pk=pk)
        data['form'] = ConsultaForm(initial={'peso':'','ca':'','data':'','rcq':'','gc':''})
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
        cir = Consulta.objects.filter(paciente=pac, evento="Cirurgia").last()
        if cir:
            numConsulta = getNumConsulta(data, cir.data)
        else:
            numConsulta = 0
        con = Consulta(evento="Consulta", numConsulta=numConsulta, peso=peso, ca=ca, rcq=rcq, gc=gc, data=data, paciente=pac)
        con.save()
        return JsonResponse(data={'message':'Consulta registrada'}, status=200)


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
            )
            p = Profissional.objects.get(pk=1)
            value.save()
            value.proficional.add(p)
            cirurgia = Consulta(
                evento="Cirurgia",
                numConsulta=0,
                peso=0,
                ca=0,
                rcq=0,
                gc=0,
                cx=paciente[8],
                data=paciente[9],
                alta=float(paciente[10]) if paciente[10] is not None else 0,
                pesoPreOp=float(paciente[0]),
                paciente=value
            )
            cirurgia.save()
            if paciente[12] != None:
                consulta = Consulta(
                        evento="Consulta",
                        numConsulta=1,
                        peso=paciente[0]-(round(cirurgia.pesoPreOp - value.getPesoIdeal(), 2)*paciente[12]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(months=1),
                        paciente=value
                )
                consulta.save()
            if paciente[13] != None:
                consulta = Consulta(
                        evento="Consulta",
                        numConsulta=2,
                        peso=paciente[0]-(round(cirurgia.pesoPreOp - value.getPesoIdeal(), 2)*paciente[13]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(months=3),
                        paciente=value
                )
                consulta.save()
            if paciente[14] != None:
                consulta = Consulta(
                        evento="Consulta",
                        numConsulta=3,
                        peso=paciente[0]-(round(cirurgia.pesoPreOp - value.getPesoIdeal(), 2)*paciente[14]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(months=6),
                        paciente=value
                )
                consulta.save()
            if paciente[15] != None:
                consulta = Consulta(
                        evento="Consulta",
                        numConsulta=4,
                        peso=paciente[0]-(round(cirurgia.pesoPreOp - value.getPesoIdeal(), 2)*paciente[15]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(months=9),
                        paciente=value
                )
                consulta.save()
            if paciente[16] != None:
                consulta = Consulta(
                        evento="Consulta",
                        numConsulta=5,
                        peso=paciente[0]-(round(cirurgia.pesoPreOp - value.getPesoIdeal(), 2)*paciente[16]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(years=1),
                        paciente=value
                )
                consulta.save()
            if paciente[17] != None:
                consulta = Consulta(
                        evento="Consulta",
                        numConsulta=6,
                        peso=paciente[0]-(round(cirurgia.pesoPreOp - value.getPesoIdeal(), 2)*paciente[17]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(years=2),
                        paciente=value
                )
                consulta.save()
            if paciente[18] != None:
                consulta = Consulta(
                        evento="Consulta",
                        numConsulta=7,
                        peso=paciente[0]-(round(cirurgia.pesoPreOp - value.getPesoIdeal(), 2)*paciente[18]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(years=3),
                        paciente=value
                )
                consulta.save()
            if paciente[19] != None:
                consulta = Consulta(
                        evento="Consulta",
                        numConsulta=8,
                        peso=paciente[0]-(round(cirurgia.pesoPreOp - value.getPesoIdeal(), 2)*paciente[19]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(years=4),
                        paciente=value
                )
                consulta.save()
            if paciente[20] != None:
                consulta = Consulta(
                        evento="Consulta",
                        numConsulta=9,
                        peso=paciente[0]-(round(cirurgia.pesoPreOp - value.getPesoIdeal(), 2)*paciente[20]),
                        ca=float(paciente[5]) if paciente[5] is not None else 0,
                        rcq=float(paciente[6]) if paciente[6] is not None else 0,
                        gc=float(paciente[7]) if paciente[7] is not None else 0,
                        data=paciente[9] + relativedelta(years=5),
                        paciente=value
                )
                consulta.save()


    return render(request, 'import.html')


def getNumConsulta(data1, data2):
    numConsuta =(data1 - data2).days
    if numConsuta < 365:
        if numConsuta // 30 == 1 or numConsuta // 30 == 0: return 1
        elif numConsuta // 30 == 3: return 2
        elif numConsuta // 30 == 6: return 3
        else: return 4
    else: return int(numConsuta / 365) + 4

"""
def dataFrame():
    d = {'pacienteId': Pacientes.objects.values_list("id", flat=True)}
    df = pd.DataFrame(data=d)
    if df is not None:
        abandonos = []
        ultimaCon = []
        Cons = np.zeros((len(df.index), 20))
        ConsMenosUlt = np.zeros((len(df.index), 4))
        for index, row in df.iterrows():
            pac = Pacientes.objects.get(pk=row["pacienteId"])
            cirurgia = Consulta.objects.filter(paciente=pac, evento="Cirurgia").last()
            if cirurgia:
                consultas = Consulta.objects.filter(paciente=pac, evento="Consulta")
                Cons[index][0] = getNumConsulta(date.today(), cirurgia.data)
                if not consultas:
                    if getNumConsulta(date.today(), cirurgia.data) > 2:
                        abandonos.append(True)
                    else:
                        abandonos.append(False)
                    ultimaCon.append(0)
                    Cons[index][1] = ultima.numConsulta
                    continue

                ultima = consultas.last()
                numConsultaHoje = getNumConsulta(date.today(), cirurgia.data)
                consultasMenosUlt = consultas.exclude(numConsulta=ultima.numConsulta)

                if (numConsultaHoje - ultima.numConsulta) >= 2:
                    abandonos.append(True)
                else:
                    abandonos.append(False)

                cons = []
                for con in consultas:
                    cons.append(float(con.peso))
                if len(cons) < 18:
                    while len(cons) != 18:
                        cons.append(0)
                Cons[index][-18:] = cons[-18:]
                Cons[index][1] = ultima.numConsulta

                cons = []
                for con in consultasMenosUlt:
                    cons.append(con.getPerdaPerc(con.peso))
                if len(cons) < 4:
                    while len(cons) != 4:
                        cons.append(0)
                ConsMenosUlt[index] = cons[-4:]

                ultimaCon.append(ultima.getPerdaPerc(ultima.peso))
            else:
                continue
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

def Regressor():
    global df
    X = np.stack(df["consultasMenosultima"], axis=0)
    y = np.array(df["ultimaCon"])

    for i in range(100):
        a = random.randint(0, 100)
        X = np.vstack([X, [a, a, a, 0]])
        y = np.append(y, [a])

    for i in range(1000):
        X = np.vstack([X, [0, 0, 0, 0]])
        y = np.append(y, [0])

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
    from sklearn.svm import SVR 

    lr = SVR()
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
    cirurgia = Consulta.objects.filter(evento="Cirurgia", paciente=pac).last()
    Cons = Consulta.objects.filter(paciente=pac, evento="Consulta")
    consultas = np.zeros(20)
    consultasSemData = np.zeros(4)
    perdaPerc = []
    consultas[0] = getNumConsulta(date.today(), cirurgia.data)
    cons = []
    for c in Cons:
        cons.append(float(c.peso))
    if len(cons) < 18:
        while len(cons) != 18:
            cons.append(0)
    consultas[-18:] = cons[-18:]
    consultas[1] = Cons.last().numConsulta
    ultimasConsultas= []
    cons = []
    datas = []
    diasParaAdicionar=0
    for c in Cons:
        cons.append(c.getPerdaPerc(c.peso))
        datas.append(c.data)
        diasParaAdicionar=90 if c.numConsulta < 9 else 365
        ultimasConsultas.append(c.getPerdaPerc(c.peso))
    if len(cons) < 4:
        while len(cons) != 4:
            cons.append(0)
    consultasSemData = cons[-4:]
    perdaPerc = ultimasConsultas[-3:]
    consultasSemData = np.array(consultasSemData)
    prob = rf.predict_proba(consultas[np.newaxis, :])
    pdp = lr.predict(consultasSemData[np.newaxis, :])[0]
    perdaPerc.append(int(round(pdp, 0)))
    datas = datas[-3:]
    datas.append((datetime(datas[-1].year, datas[-1].month, datas[-1].day) + timedelta(days=diasParaAdicionar)).date())
    lines = pd.DataFrame({
        'Paciente Previsto':perdaPerc,
    },index=datas)
    plt.figure()
    lines.plot.line()
    plt.xticks(rotation=45)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlim(datas[0], datas[-1])
    plt.savefig("./static/images/grafico.png")

    data = {}
    data['db'] = {}
    data['pac'] = pac
    data['db'] = ({'prob': round(prob[0][1])*100, 'pdp4': round(perdaPerc[-1], 2), 'id': pk})
    return render(request, 'prevPred.html', data)
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
