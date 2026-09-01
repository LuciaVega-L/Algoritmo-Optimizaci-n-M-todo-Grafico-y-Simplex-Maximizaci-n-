import re
from collections import namedtuple

def convertir_numero(numero):
    numero = numero.replace(" ", "")

    if numero in ("", "+"):
        return 1

    if numero == "-":
        return -1

    valor = float(numero)
    return int(valor) if valor.is_integer() else valor


coesFuncObj = []
restricciones = []
Restriccion = namedtuple('Restriccion', ['coeficientes', 'operador', 'independiente'])


def extraerCoeficientesPorIndice(expresion, numVariables):
    """
    Busca cada término tipo '3x1', '-x2', etc., identifica el índice
    de la variable (el número después de la x) y coloca su coeficiente
    en la posición correspondiente. Las variables no mencionadas quedan en 0.
    """
    coeficientes = [0] * numVariables

    terminos = re.findall(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)', expresion)

    for coefTexto, indiceTexto in terminos:
        indice = int(indiceTexto) - 1  # x1 -> índice 0, x2 -> índice 1, ...

        if indice < 0 or indice >= numVariables:
            raise ValueError(
                f"La variable x{indice + 1} no existe: el modelo solo "
                f"tiene {numVariables} variables de decisión."
            )

        coeficientes[indice] = convertir_numero(coefTexto)

    return coeficientes


def obtenerCoeficientesRestricciones(restriccion, numVariables):
    coeficientes = extraerCoeficientesPorIndice(restriccion, numVariables)

    resultado = re.search(
        r'(<=|>=|=|<|>)\s*(-?\d+(?:\.\d+)?)',
        restriccion
    )
    if resultado is None:
        raise ValueError(
            "No se encontró un operador (<=, >=, =) con su término "
            "independiente en la restricción."
        )

    operador = resultado.group(1)
    independiente = convertir_numero(resultado.group(2))

    restricciones.append(Restriccion(coeficientes, operador, independiente))


def obtenerCoeficientesFuncObj(funcObj, numVariables):
    coeficientes = extraerCoeficientesPorIndice(funcObj, numVariables)
    coesFuncObj.append(coeficientes)