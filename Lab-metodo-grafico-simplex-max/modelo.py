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

def obtenerCoeficientesRestricciones(restriccion, numVariables):
    coeficientes = [convertir_numero(c)
        for c in re.findall(
            r'([+-]?\s*\d*\.?\d*)\s*x\d+',
            restriccion
        )
    ]
    if len(coeficientes) != numVariables:
        raise ValueError(
            f"Se esperaban {numVariables} términos (uno por variable, "
            f"incluyendo los que valen 0), pero se encontraron {len(coeficientes)}."
        )
    resultado = re.search(
        r'(<=|>=|=|<|>)\s*(-?\d+(?:\.\d+)?)',
        restriccion
    )
    operador = resultado.group(1)
    independiente = convertir_numero(resultado.group(2))

    restricciones.append(Restriccion(coeficientes, operador, independiente))


def obtenerCoeficientesFuncObj(funcObj):
    coeficientes = [convertir_numero(c)
        for c in re.findall(
            r'([+-]?\s*\d*\.?\d*)\s*x\d+',
            funcObj
        )
    ]
    coesFuncObj.append((coeficientes))

    