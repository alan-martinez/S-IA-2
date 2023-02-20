import numpy as np
import matplotlib.pyplot as plt
import csv

class neurona:
    def __init__(self, w1, w2, b, dim):
        self.n = dim
        self.w = np.array([w1, w2]).reshape(dim, 1)
        self.b = b

    def view_output(self, x):
            y = np.dot(self.w.transpose(), x) + self.b
            if y >= 0:
                return 1
            else:
                return -1
                
file = open("weights.txt")
lines = file.readlines()
file.close()

def grafica(w_values):
    plt.clf()
    plt.scatter(X[0], X[1], color="black")
    plt.axhline(color="blue")
    plt.axvline(color="blue")
    x_values = [-3,3]
    y_values = [-(percep.w[0][0]/percep.w[1][0])*(-3) - (percep.b / percep.w[1][0]), 
                -(percep.w[0][0]/percep.w[1][0])*(3) - (percep.b / percep.w[1][0])]
    plt.plot(x_values, y_values, color="gray")

w1, w2, b = [float(value) for value in lines[0].strip().split(',')]

percep = neurona(w1, w2, b, 2)

arreglo = []

for i in range(2):
    x = np.array(np.loadtxt("entradas.csv", delimiter=',', usecols=i))
    arreglo.append(x)
X = np.array(arreglo)

y = np.array(np.loadtxt("entradas.csv", delimiter=',', usecols=2))

    
plt.clf()
plt.title("Neurona perceptrón", fontsize=20)
plt.scatter(X[0], X[1], color="black")
plt.axhline(color="blue")
plt.axvline(color="blue")
x_values = [-3,3]
y_values = [-(percep.w[0][0]/percep.w[1][0])*(-3) - (percep.b / percep.w[1][0]), -(percep.w[0][0]/percep.w[1][0])*(3) - (percep.b / percep.w[1][0])]
plt.plot(x_values, y_values, color="gray")
plt.savefig('grafica')


# for i in range(lines):
#     print(percep.predict(X[:, i]))