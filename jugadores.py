"""
Detección y seguimiento de los dos jugadores sin IA.

  1. Fondo: mediana de muchos fotogramas. Como los jugadores se mueven, la
     mediana los "borra" y queda la pista vacía (también absorbe a jueces y
     recogepelotas, que están casi quietos).
  2. Primer plano: diferencia absoluta con el fondo + umbral + morfología
     (apertura para quitar ruido, cierre para unir brazos/piernas al cuerpo).
  3. Contornos: cada mancha grande es un candidato a persona.
  4. Selección geométrica: el punto de los pies (centro del borde inferior del
     rectángulo) se proyecta a la pista real con la homografía inversa. Así las
     distancias son en metros y no dependen de la perspectiva.
  5. Se asigna un jugador a cada mitad de la pista, eligiendo el candidato más
     cercano a su posición anterior (o, al principio, a su línea de fondo).
"""
import cv2
import numpy as np

import config as C
from pista import proyectar


def calcular_fondo(ruta_video, ancho, n=C.FOTOGRAMAS_FONDO):
    cap = cv2.VideoCapture(ruta_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    muestras = []
    for i in np.linspace(0, max(total - 1, 0), min(n, max(total, 1))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            muestras.append(redimensionar(f, ancho))
    cap.release()
    if not muestras:
        raise RuntimeError("No se han podido leer fotogramas del vídeo")
    return np.median(np.stack(muestras), axis=0).astype(np.uint8)


def redimensionar(img, ancho):
    h, w = img.shape[:2]
    if w == ancho:
        return img
    return cv2.resize(img, (ancho, int(round(h * ancho / w))), interpolation=cv2.INTER_AREA)


def diferencia_absoluta(a, b):
    """|a - b| píxel a píxel. cv2.subtract satura a 0 los negativos, así que
    sumando las dos restas se obtiene el valor absoluto (en cada píxel una de
    las dos es 0, por lo que la suma no se desborda)."""
    return cv2.add(cv2.subtract(a, b), cv2.subtract(b, a))


def mascara_primer_plano(img, fondo):
    dif = diferencia_absoluta(cv2.blur(img, (3, 3)), cv2.blur(fondo, (3, 3)))
    dif = dif.max(axis=2)                       # máxima diferencia entre canales
    _, m = cv2.threshold(dif, C.UMBRAL_FONDO, 255, cv2.THRESH_BINARY)
    # Apertura (erosión + dilatación): elimina puntos sueltos de ruido.
    # Kernel en cruz (más pequeño que un cuadrado 3x3) para no borrar los pies
    # del jugador lejano, que ocupan muy pocos píxeles.
    k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    m = cv2.dilate(cv2.erode(m, k), k)
    # Cierre (dilatación + erosión): une brazos, piernas y raqueta al cuerpo
    k = np.ones((15, 9), np.uint8)
    m = cv2.erode(cv2.dilate(m, k), k)
    return m


class SeguidorJugadores:
    """Mantiene la identidad de los dos jugadores: 0 = fondo (lejano), 1 = cerca."""

    def __init__(self, H):
        self.H = H
        self.H_inv = np.linalg.inv(H)
        # Puntos de referencia ("keypoints virtuales"): 1.5 m detrás de cada fondo
        self.referencia = [np.array([C.X_CENTRO, -1.5]), np.array([C.X_CENTRO, C.LARGO + 1.5])]
        self.ultima = [None, None]          # última posición en metros
        self.perdidos = [0, 0]

    def _en_zona(self, p):
        x, y = p
        return (-C.MARGEN_LATERAL_M <= x <= C.ANCHO_DOBLES + C.MARGEN_LATERAL_M and
                -C.MARGEN_FONDO_M <= y <= C.LARGO + C.MARGEN_FONDO_M)

    def actualizar(self, mascara):
        """Devuelve [(bbox, pos_metros) o None] para cada jugador."""
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidatos = []
        for c in contornos:
            if cv2.contourArea(c) < C.AREA_MIN_JUGADOR:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h < 0.6 * w:                 # una persona es más alta que ancha (salvo estirada)
                continue
            pies = (x + w / 2, y + h)
            pos = proyectar([pies], self.H_inv)[0]
            if self._en_zona(pos):
                candidatos.append(((x, y, x + w, y + h), pos))

        resultado = [None, None]
        usados = set()
        for j in (0, 1):
            # cada jugador sólo puede estar en su mitad (con algo de holgura en la red)
            en_mitad = [(i, c) for i, c in enumerate(candidatos)
                        if i not in usados and ((c[1][1] < C.Y_RED + 1) if j == 0 else (c[1][1] > C.Y_RED - 1))]
            if not en_mitad:
                self.perdidos[j] += 1
                continue
            ref = self.ultima[j] if self.ultima[j] is not None else self.referencia[j]
            i, mejor = min(en_mitad, key=lambda ic: np.linalg.norm(ic[1][1] - ref))
            salto = np.linalg.norm(mejor[1] - ref)
            limite = C.SALTO_MAX_M * (1 + self.perdidos[j])
            if self.ultima[j] is not None and salto > limite:
                self.perdidos[j] += 1
                if self.perdidos[j] > 30:   # perdido mucho tiempo: reiniciar
                    self.ultima[j] = None
                continue
            usados.add(i)
            resultado[j] = mejor
            self.ultima[j] = mejor[1]
            self.perdidos[j] = 0
        return resultado
