import numpy as np
import matplotlib.pyplot as plt

y_axis = []

plt.ylabel('Error', fontsize=20)
plt.xlabel('Epocas', fontsize=20)
plt.title('Perceptrón unicapa', fontsize=30)

def graphError(x_coordinate, y_coordinate):
    # generar una secuencia de 32 colores distintos
    colors = plt.cm.tab20(np.linspace(0, 1, 32))
    # utilizar un color diferente para cada punto
    plt.scatter(x_coordinate, y_coordinate, c=colors[x_coordinate % 32])
    plt.pause(0.3)

class neurona:
    def __init__(self, dim, eta):   #Dimension y coeficiente de aprendizaje.
        self.n = dim
        self.eta = eta
        self.w = -1 + 2 * np.random.rand(dim, 1)
        self.b = -1 + 2 * np.random.rand()

    def predict(self, x):
        y = np.dot(self.w.transpose(), x) + self.b
        if y >= 0:
            return 1
        else:
            return -1

    def train(self, X, y, epochs):  #x = matriz de entrenamiento, y = vector con resultado esperados, epochs = épocas.
        n, m = X.shape
        average = 0
        for i in range(epochs):
            for j in range(m):
                y_pred = self.predict(X[:, j])
                if y_pred != y[j]:  #Si nuestro estimado es diferente a nuestro esperado, entrenamos.
                    self.w += self.eta*(y[j] - y_pred) * X[:, j].reshape(-1, 1)
                    self.b += self.eta*(y[j] - y_pred) 
                average += (y[j] - y_pred)**2
            average /= m
            graphError(i, average)
            y_axis.append(average)
            average = 0
        
        plt.plot(y_axis, 'g')
        y_axis.clear()
