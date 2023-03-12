import numpy as np
import matplotlib.pyplot as plt
import csv

def graph(w_values):
    plt.clf()
    #Colorear entradas, si es 1 verde si es 0 rojo
    colors = np.where(y == 1, 'green', 'red')
    plt.scatter(X[0], X[1], color=colors)

    plt.axhline(color="blue")
    plt.axvline(color="blue")
    x_values = [-3,3]
    y_values = [-(adaline.w[0][0]/adaline.w[1][0])*(-3) - (adaline.b / adaline.w[1][0]), 
                -(adaline.w[0][0]/adaline.w[1][0])*(3) - (adaline.b / adaline.w[1][0])]
    plt.plot(x_values, y_values, color="black")
    plt.pause(1)
    plt.close()

class Adaline:
    def __init__(self, dim, aprendizaje, activacion):
        self.n = dim
        self.aprendizaje = aprendizaje
        self.w = -1 + 2 * np.random.rand(dim, 1)  # x = min + (max - min)*rand()
        self.b = -1 + 2 * np.random.rand()
        self.activacion = activacion

    def clasificar(self, x):
        y = np.dot(self.w.transpose(), x) + self.b
        if self.activacion == "lineal":
            return y
        elif self.activacion == "logistica":
            return 1 / (1 + np.exp(-y))
        elif self.activacion == "sigmoidal":
            return np.tanh(y)

    def derivada_activacion(self, y):
        if self.activacion == "lineal":
            return 1
        elif self.activacion == "logistica":
            return y * (1 - y)
        elif self.activacion == "sigmoidal":
            return 1 - y ** 2

    def train(self, X, y, epocas, filename_prefix, precision=None):
        n, m = X.shape
        for i in range(epocas):
            # convertir los valores continuos de salida en valores binarios
            # comparar los vectores de salida de la red neuronal y los valores de la etiqueta de salida
            while not np.array_equal(np.sign(self.clasificar(X)), np.sign(y)):
                for j in range(m):
                    y_pred = self.clasificar(X[:, j])
                    error = y[j] - y_pred
                    delta = error * self.derivada_activacion(y_pred)
                    self.w += self.aprendizaje * delta * X[:, j].reshape(-1, 1)
                    self.b += self.aprendizaje * delta
                graph(self.w)
                if np.array_equal(np.sign(self.clasificar(X)), np.sign(y)):
                    break
                # if precision is not None and np.mean(np.allclose(self.clasificar(X), y)) >= float(precision):
                if precision is not None and np.mean(np.sign(self.clasificar(X)) == np.sign(y)) >= float(precision):  # Con esta funcion al usal la activacion logistica la precision no suele dar completamente 0 o 1
                    print(f"¡Precision del {precision*100}%!")
                    return
            plt.savefig(filename_prefix + str(i) + '.png')
            # Salir del bucle si se ha logrado una clasificación correcta
            if np.sign(self.clasificar(X)) == np.sign(y).all():
                print("¡Todas las clasificaciones son correctas!")
                break

# Obtener el numero de filas en la entrada del archivo
with open("entradas.csv") as file:
    rows = len(file.readlines())

# Obtener el numero de columnas en la entrada del archivo
with open("entradas.csv") as f:
    reader = csv.reader(f, delimiter=',')
    columns = len(next(reader))

adaline = Adaline(columns - 1, 0.1, "logistica")  # activacion por defecto es logistica
arreglo = []

for i in range(columns - 1):
    x = np.array(np.loadtxt("entradas.csv", delimiter=',', usecols=i))
    arreglo.append(x)
X = np.array(arreglo)

y = np.array(np.loadtxt("entradas.csv", delimiter=',', usecols=columns-1))

adaline.train(X, y, 10,'grafica_', 0.95) # .95 -> 95% de precision  

for i in range(rows):
    print(adaline.clasificar(X[:, i]))
