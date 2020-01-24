entradas = [1, 7, 5]
pesos = [0.8, 0.1, 0]

def soma (e, p):
    s = 0
    for i in range(3):
        s += e[i] * p[i]
    return s

def stepfunction(soma):
    result = 0
    if soma >= 1:
        result = 1
    return result

s = soma(entradas, pesos)
print(s)
r = stepfunction(s)
print(r)
