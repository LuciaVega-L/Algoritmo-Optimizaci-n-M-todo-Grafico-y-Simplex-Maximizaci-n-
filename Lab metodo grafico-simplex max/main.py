from modelo import obtenerCoeficientesRestricciones
from modelo import obtenerCoeficientesFuncObj

numVariables = int(input("Número de variables de decisión: "))
funcObj = input("Ingrese la función objetivo: ")
obtenerCoeficientesFuncObj(funcObj)
numRestricciones = int(input("Ingrese numero de restricciones: "))
for i in range(numRestricciones):
    restriccion = input(f"Ingrese la restricción {i+1}: ")
    obtenerCoeficientesRestricciones(restriccion, numVariables)

