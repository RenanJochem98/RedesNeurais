import numpy as np
from sklearn import datasets
from datetime import datetime

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

base = datasets.load_breast_cancer()
entradas = base.data
valoresSaidas = base.target

saidas = np.empty([len(valoresSaidas), 1], dtype=int)
for i in range(len(valoresSaidas)):
    saidas[i] = valoresSaidas[i]
# entradas = np.array([[0,0], [0,1],[1,0],[1,1]])
# saidas = np.array([[0],[1],[1],[0]])

quantNeuronios = 48
pesos0 = 2*np.random.random((30,quantNeuronios)) - 1
pesos1 = 2*np.random.random((quantNeuronios,1)) - 1
pesos0_inicial = pesos0
pesos1_inicial = pesos1

epocas = 1000000
taxaAprendizagem = 0.3
momento = 1 # momento serve para achar falsos minimos locais
try:
    inicio = datetime.now()
    for j in range(epocas):
        camadaEntrada = entradas

        camadaOculta = calculaAtivacao(camadaEntrada, pesos0)
        camadaSaida = calculaAtivacao(camadaOculta, pesos1)

        # gera um array com a subtracao dos valores em index iguais
        # nao eh necessario percorrer o array pq sao arrays do numpy. A lib se encarrega disso
        erroCamadaSaida = saidas - camadaSaida
        mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
        print("Erro: "+ str(mediaAbsoluta)+ " Epoca: "+ str(j))

        derivadaSaida = sigmoidDerivada(camadaSaida) # gradiente
        # deltaSaida = deltaSaida(derivadaSaida, erroCamadaSaida) #TypeError: 'numpy.ndarray' object is not callable??
        deltaSaida = erroCamadaSaida * derivadaSaida

        pesos1Transposta = pesos1.T # eh necessaria a transposta para multiplicacao de matrizes do dot product
        deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)

        deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

        # atualizacao de pesos da camada oculta, para backpropagration
        pesos1 = atualizaPesos(camadaOculta, deltaSaida, pesos1, momento, taxaAprendizagem)
        # atualizacao de pesos da camada de entrada, para backpropagration
        pesos0 = atualizaPesos(camadaEntrada, deltaCamadaOculta, pesos0, momento, taxaAprendizagem)

    fim = datetime.now()
    print()
    print("#"*20+"    RESULTADO    "+"#"*20)
    print()
    print("Epoca: "+str(j+1))
    print("Tempo: ", end=" ")
    print(fim-inicio)
    print("Erro medio: " + str(mediaAbsoluta))
    print("Pesos0 Inicial: ")
    print(pesos0_inicial)
    print()
    print("Pesos0: ")
    print(pesos0)
    print()
    print("Pesos1 Inicial:")
    print(pesos1_inicial)
    print()
    print("Pesos1:")
    print(pesos1)
    print()
    print("Saida:")
    print(camadaSaida)
    print()
    print("#"*50)
except KeyboardInterrupt:
    print("Interrompido via teclado!!")
finally:
    print("Final da execução")
