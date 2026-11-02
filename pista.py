"""
Detección de la pista y cálculo de la homografía con técnicas clásicas.

Pasos:
  1. Segmentar las líneas blancas: top-hat morfológico + umbral en HSV.
  2. Detectar segmentos rectos con la transformada de Hough probabilística.
  3. Agrupar los segmentos en líneas "horizontales" (fondo, saque) y
     "oblicuas" (laterales), fusionando las que están muy próximas.
  4. Probar combinaciones de 2 horizontales + 2 oblicuas como hipótesis de las
     4 esquinas de la pista, calcular la homografía de cada una y puntuarla
     según cuántas líneas del modelo de pista coinciden con píxeles blancos.
  5. Refinar la mejor homografía ajustando los puntos del modelo a las líneas
     reales (findHomography con RANSAC).

Si la detección automática falla, se pueden marcar las 4 esquinas a mano.
"""
import itertools
import json
import os

import cv2
import numpy as np

import config as C


# ---------------------------------------------------------------------------
# Modelo de la pista (líneas en metros)
# ---------------------------------------------------------------------------
def lineas_modelo():
    """Devuelve las líneas de la pista como pares de puntos (metros)."""
    x0, x1 = 0.0, C.ANCHO_DOBLES
    xi0, xi1 = C.PASILLO, C.ANCHO_DOBLES - C.PASILLO
    return [
        ((x0, 0), (x1, 0)),                                   # fondo lejano
        ((x0, C.LARGO), (x1, C.LARGO)),                       # fondo cercano
        ((x0, 0), (x0, C.LARGO)),                             # lateral dobles izq.
        ((x1, 0), (x1, C.LARGO)),                             # lateral dobles der.
        ((xi0, 0), (xi0, C.LARGO)),                           # lateral individual izq.
        ((xi1, 0), (xi1, C.LARGO)),                           # lateral individual der.
        ((xi0, C.Y_SAQUE_LEJANA), (xi1, C.Y_SAQUE_LEJANA)),   # saque lejana
        ((xi0, C.Y_SAQUE_CERCANA), (xi1, C.Y_SAQUE_CERCANA)), # saque cercana
        ((C.X_CENTRO, C.Y_SAQUE_LEJANA), (C.X_CENTRO, C.Y_SAQUE_CERCANA)),  # central
    ]


def puntos_modelo(paso=0.1):
    """Muestrea puntos a lo largo de todas las líneas del modelo."""
    pts = []
    for (xa, ya), (xb, yb) in lineas_modelo():
        n = max(2, int(np.hypot(xb - xa, yb - ya) / paso))
        t = np.linspace(0, 1, n)
        pts.append(np.stack([xa + t * (xb - xa), ya + t * (yb - ya)], axis=1))
    return np.concatenate(pts).astype(np.float32)


ESQUINAS_DOBLES = np.float32([[0, 0], [C.ANCHO_DOBLES, 0],
                              [C.ANCHO_DOBLES, C.LARGO], [0, C.LARGO]])


def proyectar(puntos, H):
    """Aplica una homografía a un array (N, 2) de puntos.
    En coordenadas homogéneas: [x', y', w'] = H · [x, y, 1]  ->  (x'/w', y'/w')"""
    p = np.asarray(puntos, np.float64).reshape(-1, 2)
    homogeneos = np.hstack([p, np.ones((len(p), 1))])      # (x, y, 1)
    q = homogeneos @ H.T
    return (q[:, :2] / q[:, 2:3]).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Segmentación de las líneas blancas
# ---------------------------------------------------------------------------
def mascara_lineas(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Top-hat = original - apertura (erosión + dilatación).
    # La apertura borra las estructuras claras más finas que el kernel (las
    # líneas); al restarla de la original sólo quedan esas líneas.
    k = np.ones((C.TOPHAT_KERNEL, C.TOPHAT_KERNEL), np.uint8)
    apertura = cv2.dilate(cv2.erode(gris, k), k)
    tophat = cv2.subtract(gris, apertura)
    _, m_tophat = cv2.threshold(tophat, C.TOPHAT_UMBRAL, 255, cv2.THRESH_BINARY)
    # Blanco: poca saturación y mucho brillo (filtrado por rango en HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m_blanco = cv2.inRange(hsv, np.array([0, 0, C.BLANCO_V_MIN + 1]),
                           np.array([180, C.BLANCO_S_MAX - 1, 255]))
    return cv2.bitwise_and(m_tophat, m_blanco)


# ---------------------------------------------------------------------------
# 2-3. Hough y agrupación de líneas
# ---------------------------------------------------------------------------
def _recta_homogenea(puntos):
    """Ajusta una recta a un conjunto de puntos -> coeficientes (a, b, c) de
    a·x + b·y + c = 0, por mínimos cuadrados (minimizando la distancia
    perpendicular): la dirección de la recta es la de máxima dispersión
    de los puntos, que se obtiene con la descomposición SVD."""
    p = np.asarray(puntos, np.float64)
    centro = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - centro)
    a, b = vt[1]                         # normal a la recta
    return np.array([a, b, -(a * centro[0] + b * centro[1])])


def _interseccion(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def _agrupar(segmentos, clave, tolerancia, max_grupos=8):
    """Agrupa segmentos cuya 'clave' (posición) difiere menos de 'tolerancia'."""
    if not segmentos:
        return []
    segmentos = sorted(segmentos, key=clave)
    grupos, actual = [], [segmentos[0]]
    for s in segmentos[1:]:
        if clave(s) - clave(actual[-1]) < tolerancia:
            actual.append(s)
        else:
            grupos.append(actual)
            actual = [s]
    grupos.append(actual)

    lineas = []
    for g in grupos:
        longitud = sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in g)
        pts = []
        for x1, y1, x2, y2 in g:   # muestrear cada segmento para el ajuste
            n = max(2, int(np.hypot(x2 - x1, y2 - y1) / 5))
            t = np.linspace(0, 1, n)
            pts.extend(zip(x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        lineas.append((longitud, _recta_homogenea(pts)))
    lineas.sort(key=lambda l: -l[0])                 # las más largas primero
    return [l for _, l in lineas[:max_grupos]]


def detectar_lineas(mascara):
    h, w = mascara.shape
    segs = cv2.HoughLinesP(mascara, 1, np.pi / 180, threshold=50,
                           minLineLength=int(w * 0.06), maxLineGap=12)
    if segs is None:
        return [], []
    horizontales, oblicuas = [], []
    for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(float):
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        ang = min(ang, 180 - ang)
        if ang < 8:
            horizontales.append((x1, y1, x2, y2))
        elif ang > 20:
            oblicuas.append((x1, y1, x2, y2))

    # Clave de agrupación: altura media (horizontales) o corte con y = h/2 (oblicuas)
    def y_media(s):
        return (s[1] + s[3]) / 2

    def x_en_mitad(s):
        x1, y1, x2, y2 = s
        if y2 == y1:
            return (x1 + x2) / 2
        return x1 + (h / 2 - y1) * (x2 - x1) / (y2 - y1)

    return (_agrupar(horizontales, y_media, 8),
            _agrupar(oblicuas, x_en_mitad, 12))


# ---------------------------------------------------------------------------
# 4. Búsqueda de la mejor hipótesis
# ---------------------------------------------------------------------------
def puntuar(H, mascara_dilatada, pts_modelo):
    """Coincidencias - 0.5 * fallos de las líneas del modelo proyectadas."""
    h, w = mascara_dilatada.shape
    p = np.round(proyectar(pts_modelo, H)).astype(int)
    dentro = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    if dentro.mean() < 0.7:
        return -1e9
    p = p[dentro]
    aciertos = int((mascara_dilatada[p[:, 1], p[:, 0]] > 0).sum())
    return aciertos - 0.5 * (len(pts_modelo) - aciertos)


def _es_convexo(quad):
    signos = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        u, v = b - a, c - b
        signos.append(np.sign(u[0] * v[1] - u[1] * v[0]))
    return abs(sum(signos)) == 4


def buscar_homografia(img):
    """Detección automática. Devuelve (H modelo->imagen, puntuación, máscara)."""
    h, w = img.shape[:2]
    mascara = mascara_lineas(img)
    mascara_dil = cv2.dilate(mascara, np.ones((5, 5), np.uint8))
    horizontales, oblicuas = detectar_lineas(mascara)
    pts_modelo = puntos_modelo()

    # Hipótesis: qué líneas del modelo pueden ser las 2 horizontales / 2 oblicuas
    hip_y = [(0, C.LARGO), (0, C.Y_SAQUE_CERCANA), (C.Y_SAQUE_LEJANA, C.LARGO)]
    hip_x = [(0, C.ANCHO_DOBLES), (C.PASILLO, C.ANCHO_DOBLES - C.PASILLO)]

    def x_en(l, y):
        a, b, c = l
        return -(b * y + c) / a if abs(a) > 1e-9 else 0

    # ordenar horizontales de arriba abajo y oblicuas de izquierda a derecha
    horizontales.sort(key=lambda l: _interseccion(l, np.array([1.0, 0, -w / 2]))[1]
                      if _interseccion(l, np.array([1.0, 0, -w / 2])) is not None else 0)
    oblicuas.sort(key=lambda l: x_en(l, h / 2))

    mejor = (-1e9, None)
    for hs in itertools.combinations(horizontales, 2):
        for vs in itertools.combinations(oblicuas, 2):
            quad = [_interseccion(hs[0], vs[0]), _interseccion(hs[0], vs[1]),
                    _interseccion(hs[1], vs[1]), _interseccion(hs[1], vs[0])]
            if any(q is None for q in quad):
                continue
            quad = np.float32(quad)
            if not _es_convexo(quad) or cv2.contourArea(quad) < 0.05 * w * h:
                continue
            for y_a, y_b in hip_y:
                for x_a, x_b in hip_x:
                    src = np.float32([[x_a, y_a], [x_b, y_a], [x_b, y_b], [x_a, y_b]])
                    H, _ = cv2.findHomography(src, quad)   # 4 puntos -> solución exacta
                    s = puntuar(H, mascara_dil, pts_modelo)
                    if s > mejor[0]:
                        mejor = (s, H)

    score, H = mejor
    if H is None:
        return None, 0.0, mascara
    H = refinar_homografia(H, mascara, pts_modelo)
    proporcion = (puntuar(H, mascara_dil, pts_modelo) + 0.5 * len(pts_modelo)) / (1.5 * len(pts_modelo))
    return H, proporcion, mascara


# ---------------------------------------------------------------------------
# 5. Refinamiento
# ---------------------------------------------------------------------------
def refinar_homografia(H, mascara, pts_modelo, iteraciones=3, radio=6):
    """Desplaza cada punto proyectado al centro de los píxeles blancos cercanos
    y recalcula la homografía con RANSAC."""
    h, w = mascara.shape
    ys, xs = np.nonzero(mascara)
    if len(xs) == 0:
        return H
    for _ in range(iteraciones):
        proy = proyectar(pts_modelo, H)
        origen, destino = [], []
        for pm, (px, py) in zip(pts_modelo, proy):
            x0, x1 = int(px) - radio, int(px) + radio + 1
            y0, y1 = int(py) - radio, int(py) + radio + 1
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue
            ventana = mascara[y0:y1, x0:x1]
            vy, vx = np.nonzero(ventana)
            if len(vx) < 3:
                continue
            origen.append(pm)
            destino.append((x0 + vx.mean(), y0 + vy.mean()))
        if len(origen) < 20:
            break
        H_nueva, _ = cv2.findHomography(np.float32(origen), np.float32(destino),
                                        cv2.RANSAC, 3.0)
        if H_nueva is None:
            break
        H = H_nueva
    return H


# ---------------------------------------------------------------------------
# Modo manual (plan B)
# ---------------------------------------------------------------------------
def homografia_manual(img, ruta_json):
    """El usuario marca las 4 esquinas de la pista de DOBLES con el ratón:
    sup-izq, sup-der, inf-der, inf-izq (la parte de arriba es la lejana)."""
    if os.path.exists(ruta_json):
        with open(ruta_json) as f:
            esquinas = np.float32(json.load(f))
        print(f"[pista] Esquinas cargadas de {ruta_json}")
    else:
        esquinas = []
        vista = img.copy()
        nombres = ["sup-izq", "sup-der", "inf-der", "inf-izq"]

        def click(evento, x, y, *_):
            if evento == cv2.EVENT_LBUTTONDOWN and len(esquinas) < 4:
                esquinas.append((x, y))
                cv2.circle(vista, (x, y), 5, (0, 0, 255), -1)

        ventana = "Marca las 4 esquinas de la pista de dobles (ESC para cancelar)"
        cv2.namedWindow(ventana)
        cv2.setMouseCallback(ventana, click)
        while len(esquinas) < 4:
            v = vista.copy()
            cv2.putText(v, f"Click en esquina: {nombres[len(esquinas)]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(ventana, v)
            if cv2.waitKey(20) == 27:
                raise SystemExit("Cancelado por el usuario")
        cv2.destroyAllWindows()
        esquinas = np.float32(esquinas)
        with open(ruta_json, "w") as f:
            json.dump(esquinas.tolist(), f)
        print(f"[pista] Esquinas guardadas en {ruta_json}")
    H, _ = cv2.findHomography(ESQUINAS_DOBLES, esquinas)
    return H


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def dibujar_pista(img, H, color=(0, 255, 255), grosor=2):
    for a, b in lineas_modelo():
        pa, pb = proyectar([a, b], H)
        cv2.line(img, tuple(map(int, pa)), tuple(map(int, pb)), color, grosor, cv2.LINE_AA)
    return img
