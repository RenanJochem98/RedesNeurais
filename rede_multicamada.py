import numpy as np

def calculaAtivacao(entradas, pesos):
    somaSinapse = np.dot(entradas, pesos)
    return sigmoid(somaSinapse)

def atualizaPesos(camada, delta, pesos, momento, taxaAprendizagem):
    camadaTransposta = camada.T # eh necessaria a transposta para multiplicacao de matrizes do dot product
    pesosNovo = camadaTransposta.dot(delta)
    return calculaPeso(pesos, pesosNovo, momento, taxaAprendizagem)

def calculaPeso(pesos, pesosNovo, momento, taxaAprendizagem):
    return (pesos * momento) + (pesosNovo * taxaAprendizagem)

#funcao de ativacao
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

# funcao para calculo da descida do gradiente
def sigmoidDerivada(sig):
    return sig * (1 - sig)

def deltaSaida(sigmoidDerivada, erro):
    return sigmoidDerivada * erro

# r = sigmoid(0)
entradas = np.array([[0,0], [0,1],[1,0],[1,1]])
saidas = np.array([[0],[1],[1],[0]])

pesos0 = np.array([[-0.424, -0.740,-0.961],
                    [0.358, -0.577, -0.469]])
pesos1 = np.array([[-0.017], [-0.893], [0.148]])

epocas = 10
taxaAprendizagem = 0.3
momento = 1 # momento serve para achar falsos minimos locais

for j in range(epocas):
    camadaEntrada = entradas

    camadaOculta = calculaAtivacao(camadaEntrada, pesos0)
    camadaSaida = calculaAtivacao(camadaOculta, pesos1)

    # gera um array com a subtracao dos valores em index iguais
    # nao eh necessario percorrer o array pq sao arrays do numpy. A lib se encarrega disso
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))

    derivadaSaida = sigmoidDerivada(camadaSaida) # gradiente
    deltaSaida = deltaSaida(derivadaSaida, erroCamadaSaida)

    pesos1Transposta = pesos1.T # eh necessaria a transposta para multiplicacao de matrizes do dot product
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)

    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

    # atualizacao de pesos da camada oculta, para backpropagration
    pesos1 = atualizaPesos(camadaOculta, deltaSaida, pesos1, momento, taxaAprendizagem)
    # atualizacao de pesos da camada de entrada, para backpropagration
    pesos0 = atualizaPesos(camadaEntrada, deltaCamadaOculta, pesos0, momento, taxaAprendizagem)
