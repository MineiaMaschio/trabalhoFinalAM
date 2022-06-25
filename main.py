# Fábio Franz
# Matheus Pasold
# Minéia Maschio

import pandas as pd
import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter


def calcularAcuracias(dadosTrain, rotuloTrain, dadosTeste, testRots):
    #Para o range de 1 ao tamanho da base de dados
    for i in range(1, len(dadosTeste)+1):

        #Aplica KNN
        rP = meuKnn(dadosTrain, rotuloTrain, dadosTeste, i)

        estaCorreto = rP == testRots

        numCorreto = sum(estaCorreto)

        totalNum = len(testRots)

        acuracia = np.round((numCorreto / totalNum), 2)

        print('Acurácia do KNN com k = ' + str(i))
        print(acuracia)


def getDadosRotulo(dados, rotulos, rotulo, indice):
    ret = []

    for idx in range(0, len(dados)):

        if (rotulos[idx][0] == rotulo):
            ret.append(dados[idx][indice])

    return ret


def visualizaPontos(dados, rotulos, d1, d2):
    fig, ax = plt.subplots()

    ax.scatter(getDadosRotulo(dados, rotulos, 1, d1), getDadosRotulo(dados, rotulos, 1, d2), c='red', marker='^')

    ax.scatter(getDadosRotulo(dados, rotulos, 2, d1), getDadosRotulo(dados, rotulos, 2, d2), c='blue', marker='+')

    ax.scatter(getDadosRotulo(dados, rotulos, 3, d1), getDadosRotulo(dados, rotulos, 3, d2), c='green', marker='.')


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
        listRotuloMode = []

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
    import scipy.io as scipy

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

    rotuloPrevisto = meuKnn(dadosTreinamento, rotulosTreinamento, dadosTeste, 1)

    print(rotulosTeste)
    print(rotuloPrevisto)

    estaCorreto = rotuloPrevisto == rotulosTeste

    numCorreto = sum(estaCorreto)

    totalNum = len(rotulosTeste)

    acuracia = np.round((numCorreto / totalNum), 2)

    print('Acurácia do KNN com k = 1')
    print(acuracia)

    rotuloPrevisto2 = meuKnn(dadosTreinamento, rotulosTreinamento, dadosTeste, 10)

    estaCorreto = rotuloPrevisto2 == rotulosTeste

    numCorreto = sum(estaCorreto)

    totalNum = len(rotulosTeste)

    acuracia = np.round((numCorreto / totalNum), 2)

    print('Acurácia do KNN com k = 10')
    print(acuracia)



