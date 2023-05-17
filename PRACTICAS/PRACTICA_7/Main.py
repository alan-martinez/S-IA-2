import csv
import numpy as np
from MPL import *
import matplotlib.pyplot as plt
    
def main():
    plt.figure(1)
    
    print("----------SENO CON REGRESION----------")
    function = int(input("[!]: "))
    
    trainingPatternsFileName = "Entradas.csv"
    x_funcFile = "PuntosX.csv"
    
    if function == 1:
        outputValuesFileName = "Salidas.csv"
        y_funcFile = "PuntosYSeno.csv"
        
    else:
        raise ValueError('Funcion Desconocida')
    
    
    epochs = 1500
    learning_rate = 0.1
    entries = 1  # of columns for the trainingPatternsFileName.
    neurons_in_hidden_layer = 8
    output_layer_neurons = 1
        
    net = MLP((entries, neurons_in_hidden_layer, output_layer_neurons), ('tanh', 'linear'))
              
    X = []
    y = []
    
    x_func = []
    y_func = []
    
    X.append(np.array(np.loadtxt(trainingPatternsFileName, delimiter=',', usecols=0)))
    y.append(np.array(np.loadtxt(outputValuesFileName, delimiter=',', usecols=0)))

    x_func.append(np.array(np.loadtxt(x_funcFile, delimiter=',', usecols=0)))
    y_func.append(np.array(np.loadtxt(y_funcFile, delimiter=',', usecols=0)))
    
    # Entrenamiento.

    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlim([-5, 5])
    plt.ylim([-5, 10])
        
    error_list = []
    
    for i in range(epochs):
        error, pred = net.train(X, y, 1, learning_rate)
        error_list.append(error)
        print("Epoch:", i, "Error:", error)

        if i%10 == 0:
            plt.clf()
            plt.scatter(x_func, y_func, s=40, c='#0404B4')
            plt.plot(X[0], pred[0], color='green', linewidth=3)  
            plt.show()
            plt.pause(0.2)
            plt.close()
            
            # if error < 0.03:
            if error < 0.095: #Para tanh y log.
                break

    plt.figure(2)
    plt.plot(error_list, color='red', linewidth=3)
    plt.pause(0.2)
    plt.close()
        
    results = np.array(net.predict(X)).T
    np.savetxt("Results.csv", results, delimiter=",", fmt='%.4f')
    
if __name__ == "__main__":
    main()