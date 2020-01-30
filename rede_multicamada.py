import numpy as np

def sigmoid(soma):
    # return 1 / (1 - np.exp(-soma)) # divide por zero
    return 1 / 1 - np.exp(-soma)

# r = sigmoid(0)
entradas = np.array([[0,0], [0,1],[1,0],[1,1]])
saidas = np.array([[0],[1],[1],[0]])

pesos0 = np.array([[-0.424, -0.740,-0.961],
                    [0.358, -0.577, -0.469]])
pesos1 = np.array([[-0.017], [-0.893], [0.148]])

epocas = 10
for j in range(epocas):
    camadasEntrada = entradas
    somaSinapse0 = np.dot(camadasEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dotproduct(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
