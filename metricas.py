"""
Métricas (en metros reales gracias a la homografía) y elementos visuales:
minimapa cenital, marcador de estadísticas y mapas de calor.
"""
import cv2
import numpy as np

import config as C
from pista import lineas_modelo

NOMBRES = ["Jugador fondo", "Jugador cerca"]
COLORES = [(60, 60, 255), (255, 160, 40)]          # BGR: rojo y azul


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def rellenar_y_suavizar(pos, ventana=C.SUAVIZADO_POSICION):
    """Interpola los fotogramas sin detección y aplica una media móvil."""
    pos = pos.copy()
    t = np.arange(len(pos))
    for k in range(2):
        ok = ~np.isnan(pos[:, k])
        if ok.sum() < 2:
            return pos
        pos[:, k] = np.interp(t, t[ok], pos[ok, k])
    if ventana > 1:
        nucleo = np.ones(ventana) / ventana
        borde = ventana // 2
        for k in range(2):
            relleno = np.pad(pos[:, k], borde, mode="edge")
            pos[:, k] = np.convolve(relleno, nucleo, mode="valid")
    return pos


def calcular_metricas(pos, fps):
    """pos: (N, 2) metros ya suavizados. Devuelve distancia acumulada y velocidad."""
    paso = np.zeros(len(pos))
    if len(pos) > 1:
        paso[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    vel = paso * fps
    vel[vel > C.VELOCIDAD_MAX_MS] = 0                # saltos imposibles = ruido
    paso[vel == 0] = 0
    # velocidad suavizada en ventanas de medio segundo
    v = max(1, int(fps / 2))
    vel = np.convolve(np.pad(vel, v // 2, mode="edge"), np.ones(v) / v, mode="valid")[:len(pos)]
    return np.cumsum(paso), vel


# ---------------------------------------------------------------------------
# Minimapa
# ---------------------------------------------------------------------------
class Minimapa:
    MARGEN_X, MARGEN_Y = 3.0, 5.5                    # metros alrededor de la pista

    def __init__(self, alto_px):
        self.escala = alto_px / (C.LARGO + 2 * self.MARGEN_Y)
        self.w = int((C.ANCHO_DOBLES + 2 * self.MARGEN_X) * self.escala)
        self.h = int(alto_px)
        self.base = np.full((self.h, self.w, 3), (95, 120, 60), np.uint8)
        cv2.rectangle(self.base, self.a_px(0, 0), self.a_px(C.ANCHO_DOBLES, C.LARGO), (140, 90, 50), -1)
        for a, b in lineas_modelo():
            cv2.line(self.base, self.a_px(*a), self.a_px(*b), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(self.base, self.a_px(-0.9, C.Y_RED), self.a_px(C.ANCHO_DOBLES + 0.9, C.Y_RED),
                 (200, 200, 200), 2)

    def a_px(self, x, y):
        x = min(max(x, -self.MARGEN_X), C.ANCHO_DOBLES + self.MARGEN_X)
        y = min(max(y, -self.MARGEN_Y), C.LARGO + self.MARGEN_Y)
        return (int((x + self.MARGEN_X) * self.escala), int((y + self.MARGEN_Y) * self.escala))

    def dibujar(self, img, posiciones, estelas, x0, y0, alpha=0.92):
        mapa = self.base.copy()
        for j, estela in enumerate(estelas):
            pts = [self.a_px(*p) for p in estela if not np.isnan(p[0])]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(mapa, a, b, COLORES[j], 1, cv2.LINE_AA)
        for j, p in enumerate(posiciones):
            if p is not None and not np.isnan(p[0]):
                cv2.circle(mapa, self.a_px(*p), max(4, int(self.escala * 0.45)), COLORES[j], -1, cv2.LINE_AA)
                cv2.circle(mapa, self.a_px(*p), max(4, int(self.escala * 0.45)), (255, 255, 255), 1, cv2.LINE_AA)
        zona = img[y0:y0 + self.h, x0:x0 + self.w]
        cv2.addWeighted(mapa, alpha, zona, 1 - alpha, 0, dst=zona)
        cv2.rectangle(img, (x0, y0), (x0 + self.w, y0 + self.h), (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Marcador de estadísticas
# ---------------------------------------------------------------------------
def dibujar_marcador(img, distancias, velocidades, x0, y0, escala=1.0):
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.6 * escala
    alto_linea = int(30 * escala)
    w, h = int(390 * escala), int(alto_linea * 3.3)
    zona = img[y0:y0 + h, x0:x0 + w]
    zona[:] = (zona * 0.35).astype(np.uint8)
    cv2.putText(img, "Distancia   Velocidad", (x0 + int(170 * escala), y0 + int(alto_linea * 0.9)),
                fuente, fs * 0.85, (200, 200, 200), 1, cv2.LINE_AA)
    for j in range(2):
        y = y0 + int(alto_linea * (1.9 + j))
        cv2.circle(img, (x0 + int(15 * escala), y - int(6 * escala)), int(6 * escala), COLORES[j], -1)
        cv2.putText(img, NOMBRES[j], (x0 + int(30 * escala), y), fuente, fs, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{distancias[j]:6.1f} m", (x0 + int(170 * escala), y), fuente, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{velocidades[j] * 3.6:5.1f} km/h", (x0 + int(275 * escala), y), fuente, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Mapa de calor final
# ---------------------------------------------------------------------------
def mapa_calor(posiciones_por_jugador, ruta, alto_px=700):
    mm = Minimapa(alto_px)
    paneles = []
    for j, pos in enumerate(posiciones_por_jugador):
        acumulado = np.zeros((mm.h, mm.w), np.float32)
        for p in pos:
            if not np.isnan(p[0]):
                x, y = mm.a_px(*p)
                if 0 <= x < mm.w and 0 <= y < mm.h:
                    acumulado[y, x] += 1
        k = int(mm.escala * 1.6) | 1                 # tamaño impar del suavizado
        acumulado = cv2.blur(cv2.blur(acumulado, (k, k)), (k, k))
        if acumulado.max() > 0:
            acumulado /= acumulado.max()
        # escala de color propia: amarillo (poco tiempo) -> rojo (mucho tiempo)
        color = np.zeros((mm.h, mm.w, 3), np.float32)
        color[..., 1] = 255 * (1 - acumulado)        # G
        color[..., 2] = 255                          # R
        alfa = np.clip(acumulado * 1.5, 0, 0.85)[..., None]
        panel = (mm.base * (1 - alfa) + color * alfa).astype(np.uint8)
        cabecera = np.zeros((40, mm.w, 3), np.uint8)
        cv2.putText(cabecera, NOMBRES[j], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORES[j], 2, cv2.LINE_AA)
        paneles.append(np.vstack([cabecera, panel]))
    separador = np.full((paneles[0].shape[0], 10, 3), 255, np.uint8)
    cv2.imwrite(ruta, np.hstack([paneles[0], separador, paneles[1]]))
