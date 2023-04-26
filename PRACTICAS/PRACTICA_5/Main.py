import csv
import numpy as np
from RedMulticapa import *
import matplotlib.pyplot as plt
import time

# Error graphing function.
def graphLearning(x_coordinate, y_coordinate):    
    plt.plot(x_coordinate, y_coordinate)
    plt.pause(0.2)
    
def graphError(x_coordinate, y_coordinate):    
    plt.plot(x_coordinate, y_coordinate, 'go', markersize=10)
    plt.pause(0.000000001)
    
def main():
    print("Seleccione la compuerta logica a trabajar:")
    print("1. XOR")
    print("2. XNOR")
    print("3. Salir")
    
    while True:
        try:
            option = int(input("Ingrese el número de la opción que desea: "))
            if option == 1:
                logic_gate = "xor"
                trainingPatternsFileName = "entradas.csv"
                outputValuesFileName = "xor.csv"
                break
            elif option == 2:
                logic_gate = "xnor"
                trainingPatternsFileName = "entradas.csv"
                outputValuesFileName = "xnor.csv"
                break
            elif option == 3:
                print("¡Hasta luego!")
                return
            else:
                print("Opción inválida. Por favor ingrese una opción válida.")
        except ValueError:
            print("Entrada inválida. Por favor ingrese un número.")

    
    # epochs = 10000
    epochs = 1000
    learning_rate = 0.3
    neurons_in_hidden_layer = 8
    
    file = open(trainingPatternsFileName)
    rows = len(file.readlines())
    file.close()
    
    file = open(trainingPatternsFileName,'r')
    reader = csv.reader(file,delimiter=',')
    entries = len(next(reader))
    file.close()
    
    file = open(outputValuesFileName,'r')
    reader = csv.reader(file,delimiter=',')
    output_layer_neurons = len(next(reader))
    file.close()
     
    net = RedMulticapa((entries, neurons_in_hidden_layer, output_layer_neurons), ('tanh', 'sigmoid'))
                
    patterns = []
    y = []
    
    for i in range(entries):
        x = np.array(np.loadtxt(trainingPatternsFileName, delimiter=',', usecols=i))
        patterns.append(x)
    X = np.array(patterns)
    
    for i in range(output_layer_neurons):
        y.append(np.array(np.loadtxt(outputValuesFileName, delimiter=',', usecols=i)))

    plt.figure(1)
    if logic_gate == "xor":
        plt.title("XOR", fontsize=20)
        plt.plot(0,0,'r*')
        plt.plot(0,1,'k*')
        plt.plot(1,0,'k*')
        plt.plot(1,1,'r*')
        plt.pause(5)
    elif logic_gate == "xnor":
        plt.title("XNOR", fontsize=20)
        plt.plot(0,0,'k*')
        plt.plot(0,1,'r*')
        plt.plot(1,0,'r*')
        plt.plot(1,1,'k*')
        plt.pause(5)
        
    error_list = []
    
    for i in range(epochs):
        error = net.train(X, y, 1, learning_rate)
        error_list.append(error)
        if i%10 == 0:
            graphLearning(0,0)   
            
            xx, yy = np.meshgrid(np.arange(-1, 2.1, 0.1), np.arange(-1, 2.1, 0.1))
            x_input = [xx.ravel(), yy.ravel()]
            zz = net.predict(x_input)
            zz = zz.reshape(xx.shape)
            
            plt.contourf(xx, yy, zz, alpha=0.8, cmap=plt.cm.YlOrRd)
            
            plt.xlim([-1, 2])
            plt.ylim([-1, 2])
            plt.grid()
            plt.show()
            # plt.pause(2.5)
            # plt.close()
            print("iteracion", i)
            print("error ", error)

        if error < 0.15:
            break
    
    plt.figure(2)
    
    for i in range(len(error_list)):
        graphError(i, error_list[i])
        
    results = np.array(net.predict(X)).T
    np.savetxt("Results.csv", results, delimiter=",", fmt='%.0f')
    
if __name__ == "__main__":
    main()