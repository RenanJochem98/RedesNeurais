# Este codigo eh semelhante ao arquivo perceron1.py, mas este usa o modulo numpy
import numpy as np
entradas = np.array([[0,0], [1,0], [0,1], [1,1]])
saidas = np.array([0,0,0,1])
pesos = np.array([0.0, 0.0]) # por causa do numpy, eh interessante colocar "0.0" em ez de apenas 0
taxaAprendizagem = 0.1

def stepfunction(soma):
    result = 0
    if soma >= 1:
        result = 1
    return result

def calculaSaida (registro):
    s = registro.dot(pesos) # Funcao "dot" calcula o produto escalar. (Ingles: dot Product)
    return stepfunction(s)

def treinar():
    erroTotal = 1
    while (erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print('Peso atualizado: '+str(pesos[j]))
        print("Total de erros: "+ str(erroTotal))
treinar()
