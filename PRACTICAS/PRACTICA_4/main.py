import csv
import numpy as np
import PerceptronUnicapa as percepUni

entradasFile = "Entradas1.csv"
outputValuesFileName = "SalidasDeseadas1.csv"
epochs = 10

file = open(entradasFile)
rows = len(file.readlines())
file.close()

file = open(entradasFile,'r')
reader = csv.reader(file,delimiter=',')
columns = len(next(reader))
file.close()

file = open(outputValuesFileName,'r')
reader = csv.reader(file,delimiter=',')
number_of_neurons = len(next(reader))
file.close()

neurons_array = []
      
for i in range(number_of_neurons):
    net = percepUni.neurona(columns, 0.1)
    neurons_array.append(net)
    
patterns = []
y = []

for i in range(columns):
    x = np.array(np.loadtxt(entradasFile, delimiter=',', usecols=i))
    patterns.append(x)
X = np.array(patterns)

for i in range(number_of_neurons):
    y.append(np.array(np.loadtxt(outputValuesFileName, delimiter=',', usecols=i)))

global_errors = []
for i in range(number_of_neurons):
    net.train(X, y[i], epochs)

    individual_error = []
    for i in range(rows):
        prediction = net.predict(X[:, i])
        individual_error.append(prediction)
    global_errors.append(individual_error)

global_errors = np.array(global_errors).T

np.savetxt("Results.csv", global_errors, delimiter=",", fmt='%.0f')