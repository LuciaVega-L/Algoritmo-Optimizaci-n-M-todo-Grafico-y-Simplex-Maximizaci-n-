# interfaz.py
import tkinter as tk
from tkinter import ttk, messagebox
import contextlib
import io

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modelo import obtenerCoeficientesRestricciones
from modelo import obtenerCoeficientesFuncObj
from modelo import coesFuncObj
from modelo import restricciones

from metodoGrafico import MetodoGrafico
from metodoSimplex import MetodoSimplex


class interfaz(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Laboratorio de Programación Lineal")
        self.geometry("900x650")

        self.entradasRestricciones = []
        self.numVariables = 0
        self.numRestricciones = 0
        self.metodoSeleccionado = tk.StringVar(value="grafico")

        self.contenedor = ttk.Frame(self, padding=15)
        self.contenedor.pack(fill="both", expand=True)

        self.construirFormularioInicial()

    def limpiarContenedor(self):
        for widget in self.contenedor.winfo_children():
            widget.destroy()

    # -------------------------------------------------------------
    # Pantalla 1: número de variables y de restricciones
    # -------------------------------------------------------------
    def construirFormularioInicial(self):
        self.limpiarContenedor()

        ttk.Label(self.contenedor, text="Número de variables de decisión:").grid(
            row=0, column=0, sticky="w", pady=5)
        self.entradaNumVariables = ttk.Entry(self.contenedor)
        self.entradaNumVariables.grid(row=0, column=1, pady=5)

        ttk.Label(self.contenedor, text="Número de restricciones:").grid(
            row=1, column=0, sticky="w", pady=5)
        self.entradaNumRestricciones = ttk.Entry(self.contenedor)
        self.entradaNumRestricciones.grid(row=1, column=1, pady=5)

        ttk.Button(self.contenedor, text="Continuar",
                   command=self.generarCamposModelo).grid(
            row=2, column=0, columnspan=2, pady=15)

    # -------------------------------------------------------------
    # Pantalla 2: función objetivo, restricciones y método
    # -------------------------------------------------------------
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

        ttk.Label(self.contenedor,
                  text="Formato: 2x1+3x2<=10  (operadores: <=, >=, =)",
                  font=("Arial", 9, "italic")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(self.contenedor, text="Función objetivo:").grid(
            row=1, column=0, sticky="w", pady=5)
        self.entradaFuncObj = ttk.Entry(self.contenedor, width=40)
        self.entradaFuncObj.grid(row=1, column=1, pady=5)

        filaActual = 2
        for i in range(self.numRestricciones):
            ttk.Label(self.contenedor, text=f"Restricción {i + 1}:").grid(
                row=filaActual, column=0, sticky="w", pady=5)
            entrada = ttk.Entry(self.contenedor, width=40)
            entrada.grid(row=filaActual, column=1, pady=5)
            self.entradasRestricciones.append(entrada)
            filaActual += 1

        ttk.Label(self.contenedor, text="Método:").grid(
            row=filaActual, column=0, sticky="w", pady=10)
        ttk.Radiobutton(self.contenedor, text="Gráfico", value="grafico",
                         variable=self.metodoSeleccionado).grid(
            row=filaActual, column=1, sticky="w")
        filaActual += 1
        ttk.Radiobutton(self.contenedor, text="Simplex", value="simplex",
                         variable=self.metodoSeleccionado).grid(
            row=filaActual, column=1, sticky="w")
        filaActual += 1

        ttk.Button(self.contenedor, text="Resolver",
                   command=self.resolverModelo).grid(
            row=filaActual, column=0, columnspan=2, pady=15)

        ttk.Button(self.contenedor, text="Volver",
                   command=self.construirFormularioInicial).grid(
            row=filaActual + 1, column=0, columnspan=2)

    # -------------------------------------------------------------
    # Acción: leer los campos, resolver, mostrar resultado
    # -------------------------------------------------------------
    def resolverModelo(self):
        # Limpiar listas globales de modelo.py antes de cada intento
        coesFuncObj.clear()
        restricciones.clear()

        try:
            obtenerCoeficientesFuncObj(self.entradaFuncObj.get())

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

        # Capturar los print() de consola mientras se resuelve
        salidaCapturada = io.StringIO()
        try:
            with contextlib.redirect_stdout(salidaCapturada):
                puntoOptimo, zOptimo = solver.resolver()
        except ValueError as error:
            messagebox.showerror("No se pudo resolver", str(error))
            return

        self.mostrarResultado(solver, puntoOptimo, zOptimo)

    # -------------------------------------------------------------
    # Pantalla 3: resultado (números + gráfico o iteraciones simplex)
    # -------------------------------------------------------------
    def mostrarResultado(self, solver, puntoOptimo, zOptimo):
        self.limpiarContenedor()

        textoResultado = ""
        for i in range(len(puntoOptimo)):
            textoResultado += f"x{i + 1} = {round(puntoOptimo[i], 4)}   "
        textoResultado += f"\nZ óptimo = {round(zOptimo, 4)}"

        ttk.Label(self.contenedor, text=textoResultado,
                  font=("Arial", 12, "bold")).pack(pady=10)

        if hasattr(solver, "figura"):
            canvas = FigureCanvasTkAgg(solver.figura, master=self.contenedor)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        if hasattr(solver, "historial"):
            self.mostrarIteracionesSimplex(solver.historial)

        ttk.Button(self.contenedor, text="Nuevo modelo",
                   command=self.construirFormularioInicial).pack(pady=10)

    # -------------------------------------------------------------
    # Mostrar cada iteración del simplex como tabla (Treeview)
    # -------------------------------------------------------------
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

            filaCjZj = ["Cj-Zj"] + [round(v, 3) for v in snapshot["cjMenosZj"]] + [""]
            tabla.insert("", "end", values=filaCjZj)

            ttk.Label(pestana, text=f"Z = {round(snapshot['z'], 3)}",
                      font=("Arial", 10, "bold")).pack(pady=5)


if __name__ == "__main__":
    app = interfaz()
    app.mainloop()