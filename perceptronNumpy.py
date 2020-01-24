# Este codigo eh semelhante ao arquivo perceron1.py, mas este usa o modulo numpy
import numpy as np
entradas = np.array([1, 7, 5])
pesos = np.array([0.8, 0.1, 0])

def soma (e, p):
    # Funcao dot calcula o produto escalar. (Ingles: dot Product)
    return e.dot(p)

def stepfunction(soma):
    result = 0
    if soma >= 1:
        result = 1
    return result

s = soma(entradas, pesos)
print(s)
r = stepfunction(s)
print(r)
