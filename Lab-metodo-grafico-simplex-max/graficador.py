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