import numpy as np
import matplotlib.pyplot as plt
import csv

def graph(w_values):
    plt.clf()
    for i in range(len(y)):
        if y[i] == 1:
            color = 'green'
        else:
            color = 'red'
        plt.scatter(X[0][i], X[1][i], color=color)
    plt.axhline(color="blue")
    plt.axvline(color="blue")
    x_values = [-3,3]
    y_values = [-(percep.w[0][0]/percep.w[1][0])*(-3) - (percep.b / percep.w[1][0]), 
                -(percep.w[0][0]/percep.w[1][0])*(3) - (percep.b / percep.w[1][0])]
    plt.plot(x_values, y_values, color="black")
    plt.pause(1)

# TODO dim-> Dimensiones, aprendizaje -> Coeficiente de aprendizaje

class neurona:
    def __init__(self, dim, aprendizaje):
        self.n = dim
        self.aprendizaje = aprendizaje
        self.w = -1 + 2 * np.random.rand(dim, 1)  #x = min + (max - min)*rand()
        self.b = -1 + 2 * np.random.rand()

    def predict(self, x):
        y = np.dot(self.w.transpose(), x) + self.b
        if y >= 0:
            return 1
        else:
            return -1

    # TODO x -> matriz, y ->vector de resultados esperados, epocas
    def train(self, X, y, epocas):
        n, m = X.shape
        #n = 2. m = 4 en el ejemplo de la compuerta AND, OR y XOR.
        for i in range(epocas):
            for j in range(m):
                y_pred = self.predict(X[:, j])
                # Cortar la matriz, tomando las filas (:) pero manteniendo las columnas
                if y_pred != y[j]:  #Si nuestro estimado es diferente a nuestro esperado, entrenamos.
                    self.w += self.aprendizaje*(y[j] - y_pred) * X[:, j].reshape(-1, 1)
                    self.b += self.aprendizaje*(y[j] - y_pred) 
                    graph(self.w)
                    plt.savefig(str(i) +'grafica')

file = open("entradas.csv")
rows = len(file.readlines())
file.close()

# Obtener el numero de filas en la entrada del archivo
f = open("entradas.csv",'r')
reader = csv.reader(f,delimiter=',')
columns = len(next(reader))
f.close()
# Obtener el numero de columnas en la entrada del archivo
      
percep = neurona(columns-1, 0.1)
arreglo = []

for i in range(columns-1):
    x = np.array(np.loadtxt("entradas.csv", delimiter=',', usecols=i))
    arreglo.append(x)
X = np.array(arreglo)
# Obtener el entrenamiento en el arreglo en x y la salida en y.
y = np.array(np.loadtxt("entradas.csv", delimiter=',', usecols=columns-1))

percep.train(X, y, 10) # 10 -> Numero de epocas

for i in range(rows):
    print(percep.predict(X[:, i]))
    
plt.clf()
plt.title("Neurona perceptrón con entrenamiento", fontsize=20)
plt.scatter(X[0], X[1], color="black")
plt.axhline(color="blue")
plt.axvline(color="blue")
x_values = [-3,3]
y_values = [-(percep.w[0][0]/percep.w[1][0])*(-3) - (percep.b / percep.w[1][0]), -(percep.w[0][0]/percep.w[1][0])*(3) - (percep.b / percep.w[1][0])]
plt.plot(x_values, y_values, color="gray")
# plt.savefig('grafica')