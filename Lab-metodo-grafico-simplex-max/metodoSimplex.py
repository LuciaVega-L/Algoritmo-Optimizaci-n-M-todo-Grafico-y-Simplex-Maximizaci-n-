from solver import Solver
import numpy as np


class MetodoSimplex(Solver):

    TOL = 1e-9

    def resolver(self):
        print("resolviendo el modelo utilizando el método simplex...")

        self.numVariables = len(self.coefObjetivo)
        restriccionesNormalizadas = self.normalizarRestricciones()
        self.construirTableauInicial(restriccionesNormalizadas)

        self.corregirFactibilidad()
        self.optimizar()

        puntoOptimo, zOptimo = self.extraerSolucion()

        return puntoOptimo, zOptimo

    def normalizarRestricciones(self):
        normalizadas = []

        for r in self.restricciones:
            coeficientes = list(r.coeficientes)
            operador = r.operador
            independiente = r.independiente

            if operador == '=':
                raise ValueError(
                    "El simplex clásico (sin variables artificiales) no "
                    "soporta restricciones de igualdad."
                )

            if operador == '>=':
                coeficientes = [-c for c in coeficientes]
                independiente = -independiente
                operador = '<='

            normalizadas.append((coeficientes, operador, independiente))

        return normalizadas

    def construirTableauInicial(self, restriccionesNormalizadas):
        numRestricciones = len(restriccionesNormalizadas)
        totalColumnas = self.numVariables + numRestricciones

        tabla = np.zeros((numRestricciones, totalColumnas + 1))
        costos = np.zeros(totalColumnas)
        variablesBasicas = []

        for j in range(self.numVariables):
            costos[j] = self.coefObjetivo[j]

        for i in range(numRestricciones):
            coeficientes, operador, independiente = restriccionesNormalizadas[i]

            for j in range(self.numVariables):
                tabla[i][j] = coeficientes[j]

            columnaHolgura = self.numVariables + i
            tabla[i][columnaHolgura] = 1
            variablesBasicas.append(columnaHolgura)

            tabla[i][totalColumnas] = independiente

        self.tabla = tabla
        self.costos = costos
        self.variablesBasicas = variablesBasicas
        self.totalColumnas = totalColumnas

        # Nombres legibles de cada columna: x1, x2, ..., s1, s2, ...
        self.nombresColumnas = []
        for j in range(self.numVariables):
            self.nombresColumnas.append(f"x{j + 1}")
        for i in range(numRestricciones):
            self.nombresColumnas.append(f"s{i + 1}")

        self.historial = []

    def corregirFactibilidad(self):
        iteracion = 0

        while True:
            filaPivote = self.elegirFilaNegativa()

            if filaPivote is None:
                break

            cjMenosZj, z = self.calcularCostosReducidos()
            self.registrarIteracion("Corrección de factibilidad", iteracion, cjMenosZj, z)

            columnaPivote = self.elegirColumnaDual(filaPivote, cjMenosZj)

            if columnaPivote is None:
                raise ValueError("El problema no tiene solución factible.")

            self.pivotear(filaPivote, columnaPivote)
            iteracion += 1

    def elegirFilaNegativa(self):
        filaElegida = None
        menorValor = -self.TOL

        for i in range(len(self.variablesBasicas)):
            rhs = self.tabla[i][self.totalColumnas]
            if rhs < menorValor:
                menorValor = rhs
                filaElegida = i

        return filaElegida

    def elegirColumnaDual(self, fila, cjMenosZj):
        columnaElegida = None
        mejorRazon = None

        for j in range(self.totalColumnas):
            coeficiente = self.tabla[fila][j]

            if coeficiente < -self.TOL:
                razon = cjMenosZj[j] / coeficiente

                if mejorRazon is None or razon < mejorRazon:
                    mejorRazon = razon
                    columnaElegida = j

        return columnaElegida


    def optimizar(self):
        iteracion = 0

        while True:
            cjMenosZj, z = self.calcularCostosReducidos()
            self.registrarIteracion("Optimización", iteracion, cjMenosZj, z)

            if self.esOptimo(cjMenosZj):
                break

            columnaPivote = self.elegirColumnaPivote(cjMenosZj)
            filaPivote = self.elegirFilaPivote(columnaPivote)

            if filaPivote is None:
                raise ValueError("El problema no tiene solución acotada.")

            self.pivotear(filaPivote, columnaPivote)
            iteracion += 1

    def calcularCostosReducidos(self):
        numRestricciones = len(self.variablesBasicas)

        cB = np.zeros(numRestricciones)
        for i in range(numRestricciones):
            cB[i] = self.costos[self.variablesBasicas[i]]

        zj = np.zeros(self.totalColumnas)
        for j in range(self.totalColumnas):
            suma = 0
            for i in range(numRestricciones):
                suma += cB[i] * self.tabla[i][j]
            zj[j] = suma

        cjMenosZj = self.costos - zj

        z = 0
        for i in range(numRestricciones):
            z += cB[i] * self.tabla[i][self.totalColumnas]

        return cjMenosZj, z

    def esOptimo(self, cjMenosZj):
        for valor in cjMenosZj:
            if valor > self.TOL:
                return False
        return True

    def elegirColumnaPivote(self, cjMenosZj):
        mejorIndice = 0
        mejorValor = cjMenosZj[0]

        for j in range(1, len(cjMenosZj)):
            if cjMenosZj[j] > mejorValor:
                mejorValor = cjMenosZj[j]
                mejorIndice = j

        return mejorIndice

    def elegirFilaPivote(self, columnaPivote):
        mejorFila = None
        mejorRazon = None

        for i in range(len(self.variablesBasicas)):
            coeficiente = self.tabla[i][columnaPivote]

            if coeficiente > self.TOL:
                razon = self.tabla[i][self.totalColumnas] / coeficiente

                if mejorRazon is None or razon < mejorRazon:
                    mejorRazon = razon
                    mejorFila = i

        return mejorFila

    def pivotear(self, filaPivote, columnaPivote):
        elementoPivote = self.tabla[filaPivote][columnaPivote]
        self.tabla[filaPivote] = self.tabla[filaPivote] / elementoPivote

        for i in range(len(self.variablesBasicas)):
            if i != filaPivote:
                factor = self.tabla[i][columnaPivote]
                self.tabla[i] = self.tabla[i] - factor * self.tabla[filaPivote]

        self.variablesBasicas[filaPivote] = columnaPivote

 
    def extraerSolucion(self):
        valores = np.zeros(self.totalColumnas)

        for i in range(len(self.variablesBasicas)):
            columna = self.variablesBasicas[i]
            valores[columna] = self.tabla[i][self.totalColumnas]

        puntoOptimo = tuple(valores[0:self.numVariables])

        _, zOptimo = self.calcularCostosReducidos()

        return puntoOptimo, zOptimo

    def registrarIteracion(self, fase, iteracion, cjMenosZj, z):
        nombreVariablesBasicas = []
        for indice in self.variablesBasicas:
            nombreVariablesBasicas.append(self.nombresColumnas[indice])

        snapshot = {
            "fase": fase,
            "iteracion": iteracion,
            "nombresColumnas": list(self.nombresColumnas),
            "nombreVariablesBasicas": nombreVariablesBasicas,
            "tabla": self.tabla.copy(),
            "cjMenosZj": cjMenosZj.copy(),
            "z": z,
        }
        self.historial.append(snapshot)

        print(f"\n--- {fase}, iteración {iteracion} ---")
        print("Variables básicas:", nombreVariablesBasicas)
        print("Tabla:")
        print(np.round(self.tabla, 3))
        print("Cj - Zj:", np.round(cjMenosZj, 3))
        print("Z actual:", round(z, 3))