
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

from itertools import combinations
import numpy as np
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


# solver.py
from abc import ABC, abstractmethod

class Solver(ABC):
    def __init__(self, coefObjetivo, restricciones):
        self.coefObjetivo = coefObjetivo
        self.restricciones = restricciones

    @abstractmethod
    def resolver(self):
        pass

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

# graficador.py
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def graficar(restricciones, vertices, punto_optimo, z_optimo, mostrar=True):
    puntos = np.array(vertices)

    fig, ax = plt.subplots()

    _dibujar_region_factible(ax, puntos)

    limite = _calcular_limite(puntos)
    _dibujar_restricciones(ax, restricciones, limite)
    _dibujar_punto_optimo(ax, punto_optimo, z_optimo)

    ax.set_xlim(0, limite)
    ax.set_ylim(0, limite)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

    if mostrar:
        plt.show()

    return fig

def _dibujar_region_factible(ax, puntos):
    if len(puntos) >= 3:
        hull = ConvexHull(puntos)
        orden = puntos[hull.vertices]
        ax.fill(orden[:, 0], orden[:, 1], alpha=0.3, label="Región factible")


def _calcular_limite(puntos):
    if puntos.size > 0:
        return puntos.max() * 1.2
    else:
        return 10


def _dibujar_restricciones(ax, restricciones, limite):
    x1_vals = np.linspace(0, limite, 200)

    for r in restricciones:
        a1 = r.coeficientes[0]
        a2 = r.coeficientes[1]

        if a2 != 0:
            x2_vals = (r.independiente - a1 * x1_vals) / a2
            etiqueta = f"{a1}x1 + {a2}x2 {r.operador} {r.independiente}"
            ax.plot(x1_vals, x2_vals, label=etiqueta)
        else:
            x_constante = r.independiente / a1
            ax.axvline(x_constante)


def _dibujar_punto_optimo(ax, punto_optimo, z_optimo):
    etiqueta = f"Óptimo Z={round(z_optimo, 2)}"
    ax.scatter(punto_optimo[0], punto_optimo[1], color='red',
               zorder=5, label=etiqueta)


import customtkinter as ctk
from tkinter import messagebox
import contextlib
import io
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


class interfaz(ctk.CTk):

    def __init__(self):
        super().__init__()

        # Configuración de la Ventana Principal
        self.title("Laboratorio de Programación Lineal")
        self.geometry("980x700")
        self.minsize(850, 600)
        self.configure(fg_color="#F4F6F9")  # Fondo claro pastel

        self.entradasRestricciones = []
        self.numVariables = 0
        self.numRestricciones = 0
        self.metodoSeleccionado = ctk.StringVar(value="grafico")

        # Contenedor Tarjeta Principal (Efecto elevado)
        self.card = ctk.CTkFrame(
            self,
            fg_color="#FFFFFF",
            corner_radius=16,
            border_width=1,
            border_color="#E2E8F0"
        )
        self.card.pack(fill="both", expand=True, padx=25, pady=25)

        # Contenedor interno desplazable para contenido dinámico
        self.contenedor = ctk.CTkScrollableFrame(
            self.card,
            fg_color="transparent"
        )
        self.contenedor.pack(fill="both", expand=True, padx=20, pady=20)

        self.construirFormularioInicial()

    def limpiarContenedor(self):
        for widget in self.contenedor.winfo_children():
            widget.destroy()


    def construirFormularioInicial(self):
        self.limpiarContenedor()

        # Título y Subtítulo
        lbl_titulo = ctk.CTkLabel(
            self.contenedor,
            text="Configuración del Modelo",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#1E293B"
        )
        lbl_titulo.pack(anchor="w", pady=(10, 2))

        lbl_subtitulo = ctk.CTkLabel(
            self.contenedor,
            text="Ingrese los parámetros generales para iniciar la optimización",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#64748B"
        )
        lbl_subtitulo.pack(anchor="w", pady=(0, 25))

        # Panel de Formulario
        form_frame = ctk.CTkFrame(
            self.contenedor,
            fg_color="#F8FAFC",
            corner_radius=12,
            border_width=1,
            border_color="#F1F5F9"
        )
        form_frame.pack(fill="x", pady=10, ipadx=10, ipady=10)

        # Variables
        ctk.CTkLabel(
            form_frame,
            text="Número de variables de decisión:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#334155"
        ).grid(row=0, column=0, sticky="w", padx=20, pady=15)

        self.entradaNumVariables = ctk.CTkEntry(
            form_frame,
            placeholder_text="Ej: 2",
            width=180,
            height=38,
            corner_radius=8
        )
        self.entradaNumVariables.grid(row=0, column=1, padx=20, pady=15)

        # Restricciones
        ctk.CTkLabel(
            form_frame,
            text="Número de restricciones:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#334155"
        ).grid(row=1, column=0, sticky="w", padx=20, pady=15)

        self.entradaNumRestricciones = ctk.CTkEntry(
            form_frame,
            placeholder_text="Ej: 3",
            width=180,
            height=38,
            corner_radius=8
        )
        self.entradaNumRestricciones.grid(row=1, column=1, padx=20, pady=15)

        # Botón Continuar
        btn_continuar = ctk.CTkButton(
            self.contenedor,
            text="Continuar →",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42,
            corner_radius=8,
            fg_color="#3B82F6",
            hover_color="#2563EB",
            command=self.generarCamposModelo
        )
        btn_continuar.pack(anchor="e", pady=25)


    def generarCamposModelo(self):
        try:
            self.numVariables = int(self.entradaNumVariables.get())
            self.numRestricciones = int(self.entradaNumRestricciones.get())
        except ValueError:
            messagebox.showerror("Error", "Ingrese números válidos.")
            return

        if self.numVariables <= 0 or self.numRestricciones <= 0:
            messagebox.showerror("Error", "Los valores deben ser mayores a 0.")
            return

        self.limpiarContenedor()
        self.entradasRestricciones = []

        # Encabezado
        ctk.CTkLabel(
            self.contenedor,
            text="Definición de Ecuaciones",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#1E293B"
        ).pack(anchor="w", pady=(10, 2))

        ctk.CTkLabel(
            self.contenedor,
            text="Formato aceptado: 2x1+3x2<=10  (operadores: <=, >=, =)",
            font=ctk.CTkFont(size=13),
            text_color="#64748B"
        ).pack(anchor="w", pady=(0, 20))

        # Sección Función Objetivo
        obj_frame = ctk.CTkFrame(self.contenedor, fg_color="#F8FAFC", corner_radius=10)
        obj_frame.pack(fill="x", pady=10, ipady=5)

        ctk.CTkLabel(
            obj_frame,
            text="Función objetivo:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#334155"
        ).grid(row=0, column=0, sticky="w", padx=15, pady=12)

        self.entradaFuncObj = ctk.CTkEntry(
            obj_frame,
            width=400,
            height=36,
            placeholder_text="Ej: 3x1 + 5x2",
            corner_radius=8
        )
        self.entradaFuncObj.grid(row=0, column=1, padx=15, pady=12)

        # Sección Restricciones
        rest_frame = ctk.CTkFrame(self.contenedor, fg_color="#F8FAFC", corner_radius=10)
        rest_frame.pack(fill="x", pady=10, ipady=5)

        for i in range(self.numRestricciones):
            ctk.CTkLabel(
                rest_frame,
                text=f"Restricción {i + 1}:",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#475569"
            ).grid(row=i, column=0, sticky="w", padx=15, pady=8)

            entrada = ctk.CTkEntry(
                rest_frame,
                width=400,
                height=36,
                placeholder_text=f"Ej: 2x1 + x2 <= 10",
                corner_radius=8
            )
            entrada.grid(row=i, column=1, padx=15, pady=8)
            self.entradasRestricciones.append(entrada)

        # Selección de Método
        metodo_frame = ctk.CTkFrame(self.contenedor, fg_color="#F8FAFC", corner_radius=10)
        metodo_frame.pack(fill="x", pady=10, ipady=5)

        ctk.CTkLabel(
            metodo_frame,
            text="Método de solución:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#334155"
        ).grid(row=0, column=0, sticky="w", padx=15, pady=12)

        radio_grafico = ctk.CTkRadioButton(
            metodo_frame,
            text="Gráfico",
            value="grafico",
            variable=self.metodoSeleccionado,
            font=ctk.CTkFont(size=13)
        )
        radio_grafico.grid(row=0, column=1, padx=15, pady=12)

        radio_simplex = ctk.CTkRadioButton(
            metodo_frame,
            text="Simplex",
            value="simplex",
            variable=self.metodoSeleccionado,
            font=ctk.CTkFont(size=13)
        )
        radio_simplex.grid(row=0, column=2, padx=15, pady=12)

        # Botones de Acción
        btn_box = ctk.CTkFrame(self.contenedor, fg_color="transparent")
        btn_box.pack(fill="x", pady=20)

        btn_volver = ctk.CTkButton(
            btn_box,
            text="← Volver",
            fg_color="#E2E8F0",
            hover_color="#CBD5E1",
            text_color="#334155",
            height=40,
            corner_radius=8,
            command=self.construirFormularioInicial
        )
        btn_volver.pack(side="left")

        btn_resolver = ctk.CTkButton(
            btn_box,
            text="Resolver Modelo",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669",
            height=40,
            corner_radius=8,
            command=self.resolverModelo
        )
        btn_resolver.pack(side="right")


    def resolverModelo(self):
        coesFuncObj.clear()
        restricciones.clear()

        try:
            obtenerCoeficientesFuncObj(self.entradaFuncObj.get(), self.numVariables)

            for entrada in self.entradasRestricciones:
                obtenerCoeficientesRestricciones(entrada.get(), self.numVariables)

        except ValueError as error:
            messagebox.showerror("Error en los datos", str(error))
            return

        coefObjetivo = coesFuncObj[0]

        if self.metodoSeleccionado.get() == "grafico":
            solver = MetodoGrafico(coefObjetivo, restricciones)
        else:
            solver = MetodoSimplex(coefObjetivo, restricciones)

        salidaCapturada = io.StringIO()
        try:
            with contextlib.redirect_stdout(salidaCapturada):
                puntoOptimo, zOptimo = solver.resolver()
        except ValueError as error:
            messagebox.showerror("No se pudo resolver", str(error))
            return

        self.mostrarResultado(solver, puntoOptimo, zOptimo)

    def mostrarResultado(self, solver, puntoOptimo, zOptimo):
        self.limpiarContenedor()

        # Título
        ctk.CTkLabel(
            self.contenedor,
            text="Resultados de la Optimización",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#1E293B"
        ).pack(anchor="w", pady=(10, 15))

        resumen_box = ctk.CTkFrame(
            self.contenedor,
            fg_color="#EFF6FF",
            border_width=1,
            border_color="#BFDBFE",
            corner_radius=12
        )
        resumen_box.pack(fill="x", pady=(0, 20), ipady=10, ipadx=10)

        vars_str = ",  ".join([f"x{i+1} = {round(val, 4)}" for i, val in enumerate(puntoOptimo)])

        ctk.CTkLabel(
            resumen_box,
            text=f"Solución Óptima:  {vars_str}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#1E40AF"
        ).pack(anchor="w", padx=15, pady=2)

        ctk.CTkLabel(
            resumen_box,
            text=f"Valor Óptimo Z = {round(zOptimo, 4)}",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#1D4ED8"
        ).pack(anchor="w", padx=15, pady=4)

        if hasattr(solver, "figura"):
            canvas_frame = ctk.CTkFrame(self.contenedor, fg_color="#FFFFFF", corner_radius=10)
            canvas_frame.pack(fill="both", expand=True, pady=10)

            canvas = FigureCanvasTkAgg(solver.figura, master=canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        if hasattr(solver, "historial"):
            self.mostrarIteracionesSimplex(solver.historial)

        # Botón Nuevo Modelo
        ctk.CTkButton(
            self.contenedor,
            text="← Iniciar Nuevo Modelo",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#3B82F6",
            hover_color="#2563EB",
            height=40,
            corner_radius=8,
            command=self.construirFormularioInicial
        ).pack(anchor="w", pady=20)


    def mostrarIteracionesSimplex(self, historial):
        notebook = ttk.Notebook(self.contenedor)
        notebook.pack(fill="both", expand=True, pady=10)

        for snapshot in historial:
            pestana = ttk.Frame(notebook)
            titulo = f"{snapshot['fase']} #{snapshot['iteracion']}"
            notebook.add(pestana, text=titulo)

            columnas = ["Base"] + snapshot["nombresColumnas"] + ["RHS"]

            tabla = ttk.Treeview(pestana, columns=columnas, show="headings", height=8)
            for col in columnas:
                tabla.heading(col, text=col)
                tabla.column(col, width=70, anchor="center")
            tabla.pack(fill="both", expand=True)

            numFilas = snapshot["tabla"].shape[0]
            for i in range(numFilas):
                nombreBase = snapshot["nombreVariablesBasicas"][i]
                valores = [round(v, 3) for v in snapshot["tabla"][i]]
                fila = [nombreBase] + valores
                tabla.insert("", "end", values=fila)

            valoresCjMenosZj = [round(v, 3) for v in snapshot["cjMenosZj"]]
            filaZ = ["Z"] + valoresCjMenosZj + [round(snapshot["z"], 3)]
            tabla.insert("", "end", values=filaZ, tags=("filaZ",))

            tabla.tag_configure("filaZ", background="#fff3cd")


if __name__ == "__main__":
    app = interfaz()
    app.mainloop()

    