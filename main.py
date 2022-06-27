# Fábio Franz
# Matheus Pasold
# Minéia Maschio

import pandas as pd
import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def calcularAcuracias(dadosTrain, rotuloTrain, dadosTeste, testRots):
    #Para o range de 1 ao tamanho da base de dados
    for i in range(1, 15):

        #Aplica KNN
        rP = meuKnn(dadosTrain, rotuloTrain, dadosTeste, i)

        acuracia = np.round(accuracy_score(testRots, rP), 2)

        print('Acurácia do KNN com k = ' + str(i))
        print(acuracia)


def getDadosRotulo(dados, rotulos, rotulo, indice):
    ret = []

    for idx in range(0, len(dados)):

        if (rotulos[idx] == rotulo):
            ret.append(dados[idx][indice])

    return ret


def visualizaPontos(dados, rotulos, d1, d2):
    fig, ax = plt.subplots()

    ax.scatter(getDadosRotulo(dados, rotulos, 'mask', d1), getDadosRotulo(dados, rotulos, 'mask', d2), c='red', marker='^')

    ax.scatter(getDadosRotulo(dados, rotulos, 'no-mask', d1), getDadosRotulo(dados, rotulos, 'no-mask', d2), c='blue', marker='+')

    plt.show()

def normalizacao(dados):
    #Minímos de cada atributo
    min = np.amin(dados, axis=0)

    # Máximos de cada atributo
    max = np.amax(dados, axis=0)

    #Para cada atributo
    for i in range(len(dados[0])):

        #Para cada dado
        for x in range(len(dados)):

            #Divisor
            divisor = dados[x][i] - min[i]

            #Dividendo
            dividendo = max[i] - min[i]

            #Resultado é salvo na própria matriz
            dados[x][i] = divisor / dividendo

def dist(dadosTrain, dadosTeste, rotuloTrain):
    #Lista de distâncias
    listDist = []

    #Para cada dado de teste
    for i in range(len(dadosTrain)):
        soma = 0
        d = []

        #Para cada atributo do teste calcular distancia euclidiana
        for x in range(len(dadosTeste)):
            soma += pow((dadosTrain[i][x] - dadosTeste[x]),2)

        #Adiciona distância
        d.append(math.sqrt(soma))

        #Adiciona rotulo
        d.append(rotuloTrain[i])

        #Adiciona aos resultados
        listDist.append(d)

    return listDist

def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
    #Lista de resultados
    listResult = []

    #Para cada exemplo de teste
    for x in range(len(dadosTeste)):
        listRotulos = []

        #Calcule a distância entre o exemplo de teste e os dados de treinamento
        listDist = dist(dadosTrain, dadosTeste[x], rotuloTrain)

        # Ordenar lista
        listSorted = sorted(listDist, key=itemgetter(0))

        # Cria uma nova lista com k sendo a quantidade de itens
        listK = listSorted[:k]

        # Pega apenas os rotulos
        for x in range(len(listK)):
            listRotulos.append(listK[x][1])

        #Calcula a moda e adiciona ao resultado
        listResult.append(statistics.mode(listRotulos))

    return listResult

if __name__ == '__main__':
    # Ler aquivo de dados
    baseTeste = pd.read_csv("test/_annotations.csv", header=None)
    baseTreinamento = pd.read_csv("train/_annotations.csv", header=None)

    # Extrair dados e rotulos da base de treinamento
    dadosTreinamento = baseTreinamento.iloc[1:, 4:8].values
    rotulosTreinamento = baseTreinamento.iloc[1:, 3].values

    # Extrair dados e rotulos da base de teste
    dadosTeste = baseTeste.iloc[1:, 4:8].values
    rotulosTeste = baseTeste.iloc[1:, 3].values

    # Converter dados de treinamento para float
    dadosTreinamento[:, 0:4] = dadosTreinamento[:, 0:4].astype(float)

    # Converter dados de teste para float
    dadosTeste[:, 0:4] = dadosTeste[:, 0:4].astype(float)

    # Calcular kNN com k == 1
    rotuloPrevisto = meuKnn(dadosTreinamento, rotulosTreinamento, dadosTeste, 1)

    # Cálculo da acurácia
    acuracia = np.round(accuracy_score(rotulosTeste,rotuloPrevisto),2)
    print('Acurácia do KNN com k = 1')
    print(acuracia)

    #visualizaPontos(dadosTeste, rotuloPrevisto, 1, 2)

    # Calcular kNN para k de 1 a 15
    calcularAcuracias(dadosTreinamento, rotulosTreinamento, dadosTeste, rotulosTeste)

    # Calculo de kNN com melhor acurácia
    rotuloPrevisto = meuKnn(dadosTreinamento, rotulosTreinamento, dadosTeste, 7)

    # Matriz de confusão para melhor acuracia
    cf_matrix = confusion_matrix(rotulosTeste, rotuloPrevisto)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Matriz de Confusão\n\n');

    ax.xaxis.set_ticklabels(['Máscara', 'Não máscara'])
    ax.yaxis.set_ticklabels(['Máscara', 'Não máscara'])

    plt.show()

    # Classificação
    report = classification_report(rotulosTeste, rotuloPrevisto, output_dict=True)

    # Parâmetro classification_report
    print('\nClassification Report')
    print(report)



