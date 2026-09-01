from solver import Solver
from itertools import combinations
from graficador import graficar
import numpy as np

class MetodoGrafico(Solver):
    def resolver(self):
        print("resolviendo el modelo utilizando el método gráfico...")

        if len(self.coefObjetivo) != 2:
            raise ValueError("El método gráfico solo es aplicable a problemas con dos variables de decisión.")

        lineas = self.construirLineas()
        intersecciones = self.hallarIntersecciones(lineas)
        vertices = self.filtrarFactibles(intersecciones)
        if len(vertices) == 0:
            raise ValueError("No se encontró una región factible.")
        puntoOptimo, zOptimo = self.evaluarObjetivo(vertices)

        self.figura = graficar(self.restricciones, vertices, puntoOptimo, zOptimo, mostrar=False)

        return puntoOptimo, zOptimo

    def construirLineas(self):
        lineas = [(r.coeficientes[0], r.coeficientes[1], r.independiente)
                  for r in self.restricciones]

        # No negatividad: x1 = 0 y x2 = 0
        lineas.append((1, 0, 0))  # eje x2 (x1 = 0)
        lineas.append((0, 1, 0))  # eje x1 (x2 = 0)

        return lineas
    
    def hallarIntersecciones(self, lineas):
        """Resuelve el sistema 2x2 para cada par de líneas."""
        puntos = []
        for (x1, x2, c1), (x3, x4, c2) in combinations(lineas, 2):
            A = np.array([[x1, x2], [x3, x4]], dtype=float)
            b = np.array([c1, c2], dtype=float)
            try:
                punto = np.linalg.solve(A, b)
                puntos.append(punto)
            except np.linalg.LinAlgError:
                continue  
        return puntos
    
    def esFactible(self, punto, tol=1e-9):
        x1 = punto[0]
        x2 = punto[1]

        # No negatividad
        if x1 < -tol:
            return False
        if x2 < -tol:
            return False

        # Revisar cada restricción original
        for r in self.restricciones:
            a1 = r.coeficientes[0]
            a2 = r.coeficientes[1]
            valor = a1 * x1 + a2 * x2

            if r.operador == '<=':
                if valor > r.independiente + tol:
                    return False

            elif r.operador == '>=':
                if valor < r.independiente - tol:
                    return False

            elif r.operador == '=':
                diferencia = abs(valor - r.independiente)
                if diferencia > tol:
                    return False

        # Si pasó todas las pruebas, es factible
        return True

    def yaExiste(self, punto, lista_puntos, tol=1e-6):
        for p in lista_puntos:
            distancia = np.linalg.norm(np.array(punto) - np.array(p))
            if distancia < tol:
                return True
        return False

    def filtrarFactibles(self, puntos):
        vertices = []

        for p in puntos:
            if self.esFactible(p):
                if not self.yaExiste(p, vertices):
                    vertices.append(p)

        return vertices

    def evaluarObjetivo(self, vertices):
        c1 = self.coefObjetivo[0]
        c2 = self.coefObjetivo[1]

        valoresZ = []
        for v in vertices:
            x1 = v[0]
            x2 = v[1]
            z = c1 * x1 + c2 * x2
            valoresZ.append(z)

        mejorIndice = 0
        mejorZ = valoresZ[0]

        for i in range(1, len(valoresZ)):
            if valoresZ[i] > mejorZ:
                mejorZ = valoresZ[i]
                mejorIndice = i

        puntoOptimo = vertices[mejorIndice]
        zOptimo = mejorZ

        return puntoOptimo, zOptimo

    