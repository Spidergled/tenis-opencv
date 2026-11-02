"""
Detección y seguimiento de la pelota sin IA.

  1. Diferencia de TRES fotogramas consecutivos (t-1, t, t+1): un píxel es
     candidato si cambia tanto de t-1 a t como de t a t+1. Así se queda sólo
     la posición de la pelota en t, y no sus "fantasmas" en t-1 y t+1.
  2. La pelota es más clara que la pista: se exige un brillo mínimo (V en HSV).
  3. Se descartan las zonas de los jugadores y lo que queda fuera de una zona
     ampliada alrededor de la pista (público, marcadores...).
  4. Filtrado de manchas por área y tamaño (la pelota es pequeña).
  5. Coherencia temporal (fuera de línea): una detección se confirma si forma
     un "triplete" con detecciones de los fotogramas vecinos con velocidad casi
     constante. Después se enlaza la trayectoria, se eliminan los saltos
     imposibles y se interpolan los huecos cortos.
"""
import cv2
import numpy as np

import config as C
from jugadores import diferencia_absoluta
from pista import proyectar


def zona_valida(H, forma):
    """Máscara de la zona donde puede estar la pelota: la pista ampliada.
    Se amplía mucho por arriba porque la pelota vuela por encima del fondo lejano."""
    h, w = forma
    poligono = proyectar([[-5, -9], [C.ANCHO_DOBLES + 5, -9],
                          [C.ANCHO_DOBLES + 5, C.LARGO + 6], [-5, C.LARGO + 6]], H)
    # Un píxel está dentro del cuadrilátero (convexo) si queda al mismo lado
    # de sus 4 lados: se comprueba con el signo del producto vectorial.
    ys, xs = np.mgrid[0:h, 0:w]
    dentro = np.ones((h, w), bool)
    signo = None
    for i in range(4):
        (x0, y0), (x1, y1) = poligono[i], poligono[(i + 1) % 4]
        lado = (x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0)
        if signo is None:
            signo = np.sign((x1 - x0) * (poligono[(i + 2) % 4][1] - y0) -
                            (y1 - y0) * (poligono[(i + 2) % 4][0] - x0))
        dentro &= (lado * signo) >= 0
    return dentro.astype(np.uint8) * 255


def candidatos(gris_prev, gris, gris_sig, hsv, zona, cajas_jugadores):
    _, m1 = cv2.threshold(diferencia_absoluta(gris, gris_prev), C.UMBRAL_DIF_PELOTA, 255, cv2.THRESH_BINARY)
    _, m2 = cv2.threshold(diferencia_absoluta(gris_sig, gris), C.UMBRAL_DIF_PELOTA, 255, cv2.THRESH_BINARY)
    _, m3 = cv2.threshold(hsv[..., 2], C.V_MIN_PELOTA, 255, cv2.THRESH_BINARY)
    m = cv2.bitwise_and(cv2.bitwise_and(m1, m2), cv2.bitwise_and(m3, zona))
    for x0, y0, x1, y1 in cajas_jugadores:       # quitar a los jugadores
        mx, my = int(0.15 * (x1 - x0)), int(0.1 * (y1 - y0))
        m[max(0, y0 - my):y1 + my, max(0, x0 - mx):x1 + mx] = 0
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))

    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    salida = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        mancha = m[y:y + h, x:x + w]
        area = np.count_nonzero(mancha)        # nº de píxeles de la mancha
        if (C.AREA_MIN_PELOTA <= area <= C.AREA_MAX_PELOTA and
                w <= C.LADO_MAX_PELOTA and h <= C.LADO_MAX_PELOTA):
            ys, xs = np.nonzero(mancha)        # centro = media de sus píxeles
            salida.append((x + xs.mean(), y + ys.mean()))
    return salida


def construir_trayectoria(cands):
    """cands: lista (por fotograma) de listas de (x, y). Devuelve array (N, 2)
    con la posición de la pelota o NaN si no se conoce."""
    N = len(cands)
    tray = np.full((N, 2), np.nan)

    # --- 1. Confirmar detecciones mediante tripletes de velocidad constante ---
    confirmados = [dict() for _ in range(N)]     # índice candidato -> error del triplete
    for t in range(1, N - 1):
        for ia, a in enumerate(cands[t - 1]):
            for ib, b in enumerate(cands[t]):
                v1 = np.subtract(b, a)
                if not (1 < np.hypot(*v1) < 60):
                    continue
                for ic, c in enumerate(cands[t + 1]):
                    v2 = np.subtract(c, b)
                    err = np.hypot(*(v2 - v1))
                    if err < 6 + 0.3 * np.hypot(*v1):
                        for tt, ii in ((t - 1, ia), (t, ib), (t + 1, ic)):
                            confirmados[tt][ii] = min(err, confirmados[tt].get(ii, 1e9))

    # --- 2. Enlazar: en cada fotograma, el confirmado más cercano a la predicción ---
    ultimo_t, ultimo_p, vel = None, None, np.zeros(2)
    for t in range(N):
        if not confirmados[t]:
            continue
        opciones = [(i, np.array(cands[t][i])) for i in confirmados[t]]
        if ultimo_p is not None and t - ultimo_t <= 10:
            pred = ultimo_p + vel * (t - ultimo_t)
            i, p = min(opciones, key=lambda o: np.linalg.norm(o[1] - pred))
            if np.linalg.norm(p - pred) > 40 + 15 * (t - ultimo_t):
                # muy lejos de lo esperado: sólo aceptarlo si es un triplete muy bueno
                i, p = min(opciones, key=lambda o: confirmados[t][o[0]])
                if confirmados[t][i] > 3:
                    continue
        else:
            i, p = min(opciones, key=lambda o: confirmados[t][o[0]])
        if ultimo_p is not None and t - ultimo_t <= 10:
            vel = (p - ultimo_p) / (t - ultimo_t)
        else:
            vel = np.zeros(2)
        tray[t] = p
        ultimo_t, ultimo_p = t, p

    # --- 3. Interpolar huecos cortos ---
    validos = np.where(~np.isnan(tray[:, 0]))[0]
    for a, b in zip(validos[:-1], validos[1:]):
        if 1 < b - a <= C.HUECO_MAX_INTERPOLAR + 1:
            for t in range(a + 1, b):
                f = (t - a) / (b - a)
                tray[t] = tray[a] * (1 - f) + tray[b] * f
    return tray
