import customtkinter as ctk
from tkinter import messagebox
import contextlib
import io

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from modelo import obtenerCoeficientesRestricciones
from modelo import obtenerCoeficientesFuncObj
from modelo import coesFuncObj
from modelo import restricciones

from metodoGrafico import MetodoGrafico
from metodoSimplex import MetodoSimplex

# Configuración global de CustomTkinter (Modo Claro y Paleta Pastel Azul)
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

    # -------------------------------------------------------------
    # Pantalla 1: número de variables y de restricciones
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Acción: leer los campos, resolver, mostrar resultado
    # -------------------------------------------------------------
    def resolverModelo(self):
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

        # Título
        ctk.CTkLabel(
            self.contenedor,
            text="Resultados de la Optimización",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#1E293B"
        ).pack(anchor="w", pady=(10, 15))

        # Tarjeta Resumen en Tono Pastel Suave (Azul/Verde)
        resumen_box = ctk.CTkFrame(
            self.contenedor,
            fg_color="#EFF6FF",
            border_width=1,
            border_color="#BFDBFE",
            corner_radius=12
        )
        resumen_box.pack(fill="x", pady=(0, 20), ipady=10, ipadx=10)

        # Construcción texto de variables
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

        # Gráfico (Método Gráfico)
        if hasattr(solver, "figura"):
            canvas_frame = ctk.CTkFrame(self.contenedor, fg_color="#FFFFFF", corner_radius=10)
            canvas_frame.pack(fill="both", expand=True, pady=10)

            canvas = FigureCanvasTkAgg(solver.figura, master=canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Tablas de Iteraciones (Método Simplex)
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

    # -------------------------------------------------------------
    # Mostrar cada iteración del simplex como pestañas (CTkTabview)
    # -------------------------------------------------------------
    def mostrarIteracionesSimplex(self, historial):
        tabview = ctk.CTkTabview(self.contenedor, height=250)
        tabview.pack(fill="both", expand=True, pady=10)

        for snapshot in historial:
            titulo_pestana = f"{snapshot['fase']} #{snapshot['iteracion']}"
            tab = tabview.add(titulo_pestana)

            columnas = ["Base"] + snapshot["nombresColumnas"] + ["RHS"]

            # Contenedor de la tabla simulada
            tabla_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
            tabla_frame.pack(fill="both", expand=True)

            # Encabezado
            for col_idx, text_col in enumerate(columnas):
                lbl_enc = ctk.CTkLabel(
                    tabla_frame,
                    text=text_col,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    fg_color="#E2E8F0",
                    corner_radius=4,
                    width=75,
                    height=28
                )
                lbl_enc.grid(row=0, column=col_idx, padx=2, pady=2)

            # Filas de datos
            numFilas = snapshot["tabla"].shape[0]
            for i in range(numFilas):
                nombreBase = snapshot["nombreVariablesBasicas"][i]
                valores = [round(v, 3) for v in snapshot["tabla"][i]]
                fila = [nombreBase] + valores

                for col_idx, val in enumerate(fila):
                    lbl_val = ctk.CTkLabel(
                        tabla_frame,
                        text=str(val),
                        font=ctk.CTkFont(size=11),
                        fg_color="#F8FAFC",
                        corner_radius=4,
                        width=75,
                        height=26
                    )
                    lbl_val.grid(row=i + 1, column=col_idx, padx=2, pady=2)

            # Fila Cj - Zj
            filaCjZj = ["Cj-Zj"] + [round(v, 3) for v in snapshot["cjMenosZj"]] + [""]
            for col_idx, val in enumerate(filaCjZj):
                lbl_cj = ctk.CTkLabel(
                    tabla_frame,
                    text=str(val),
                    font=ctk.CTkFont(size=11, weight="bold"),
                    fg_color="#FEF3C7",  # Tono amarillo pastel para destacar la fila de control
                    text_color="#92400E",
                    corner_radius=4,
                    width=75,
                    height=26
                )
                lbl_cj.grid(row=numFilas + 1, column=col_idx, padx=2, pady=2)

            # Z de la iteración
            ctk.CTkLabel(
                tab,
                text=f"Valor de Z = {round(snapshot['z'], 3)}",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#0F172A"
            ).pack(anchor="e", pady=8, padx=10)


if __name__ == "__main__":
    app = interfaz()
    app.mainloop()