# Este codigo eh semelhante ao arquivo perceron1.py, mas este usa o modulo numpy
import numpy as np
entradas = np.array([[0,0], [1,0], [0,1], [1,1]])
saidas = np.array([0,0,0,1])
pesos = np.array([0.0, 0.0, 0.0]) # por causa do numpy, eh interessante colocar "0.0" em ez de apenas 0
taxaAprendizagem = 0.1
def soma (e, p):
    # Funcao dot calcula o produto escalar. (Ingles: dot Product)
    return e.dot(p)

def stepfunction(soma):
    result = 0
    if soma >= 1:
        result = 1
    return result

def treinar():
    return 0
