# main.py
from modelo import obtenerCoeficientesRestricciones
from modelo import obtenerCoeficientesFuncObj
from modelo import coesFuncObj
from modelo import restricciones

from metodoGrafico import MetodoGrafico
from metodoSimplex import MetodoSimplex


def pedirMetodo():
    while True:
        metodo = input("¿Qué método desea usar? (grafico/simplex): ").strip().lower()
        if metodo in ("grafico", "simplex"):
            return metodo
        print("Opción inválida. Escriba 'grafico' o 'simplex'.")


def pedirDatosModelo():
    numVariables = int(input("Número de variables de decisión: "))

    funcObj = input("Ingrese la función objetivo (ejemplo: 3x1 + 5x2): ")
    obtenerCoeficientesFuncObj(funcObj, numVariables)

    numRestricciones = int(input("Ingrese numero de restricciones: "))
    for i in range(numRestricciones):
        restriccion = input(f"Ingrese la restricción {i + 1}: ")
        obtenerCoeficientesRestricciones(restriccion, numVariables)


def main():
    pedirDatosModelo()

    metodo = pedirMetodo()

    coefObjetivo = coesFuncObj[0]

    if metodo == "grafico":
        solver = MetodoGrafico(coefObjetivo, restricciones)
    else:
        solver = MetodoSimplex(coefObjetivo, restricciones)

    try:
        puntoOptimo, zOptimo = solver.resolver()
    except ValueError as error:
        print(f"\nNo se pudo resolver el modelo: {error}")
        return

    print("\n--- Resultado ---")
    for i in range(len(puntoOptimo)):
        print(f"x{i + 1} = {round(puntoOptimo[i], 4)}")
    print(f"Z óptimo = {round(zOptimo, 4)}")


if __name__ == "__main__":
    main()