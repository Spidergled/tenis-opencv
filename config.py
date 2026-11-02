"""
Parámetros globales del sistema.

Todas las medidas de la pista están en metros (reglamento ITF).
Sistema de coordenadas de la pista (vista cenital):
  - origen (0, 0) en la esquina superior izquierda de la pista de dobles
    (la línea de fondo LEJANA a la cámara),
  - eje X a lo ancho (0 -> 10.97 m),
  - eje Y a lo largo (0 -> 23.77 m), creciendo hacia la cámara.
"""

# ---------------------------------------------------------------------------
# Medidas oficiales de la pista (metros)
# ---------------------------------------------------------------------------
ANCHO_DOBLES = 10.97
ANCHO_INDIVIDUAL = 8.23
LARGO = 23.77
DIST_SAQUE = 6.40           # distancia de la red a la línea de saque
PASILLO = (ANCHO_DOBLES - ANCHO_INDIVIDUAL) / 2     # 1.37 m
Y_RED = LARGO / 2                                   # 11.885 m
Y_SAQUE_LEJANA = Y_RED - DIST_SAQUE                 # 5.485 m
Y_SAQUE_CERCANA = Y_RED + DIST_SAQUE                # 18.285 m
X_CENTRO = ANCHO_DOBLES / 2

# ---------------------------------------------------------------------------
# Procesado
# ---------------------------------------------------------------------------
ANCHO_TRABAJO = 960         # se reduce cada fotograma a este ancho para ir más rápido
FOTOGRAMAS_FONDO = 60       # nº de fotogramas para estimar el fondo (mediana)

# --- Detección de líneas de la pista ---
TOPHAT_KERNEL = 11          # tamaño del elemento estructurante del top-hat
TOPHAT_UMBRAL = 25          # brillo mínimo de una línea respecto a su entorno
BLANCO_S_MAX = 90           # saturación máxima (HSV) para considerar un píxel blanco
BLANCO_V_MIN = 140          # brillo mínimo (HSV) para considerar un píxel blanco

# --- Jugadores ---
UMBRAL_FONDO = 30           # diferencia mínima con el fondo para ser "primer plano"
AREA_MIN_JUGADOR = 120      # área mínima del contorno (px, a ANCHO_TRABAJO)
MARGEN_LATERAL_M = 4.0      # zona válida alrededor de la pista (metros)
MARGEN_FONDO_M = 8.0
SALTO_MAX_M = 1.5           # desplazamiento máximo creíble entre dos fotogramas

# --- Pelota ---
UMBRAL_DIF_PELOTA = 15      # diferencia mínima entre fotogramas consecutivos
V_MIN_PELOTA = 110          # la pelota es más clara que la pista
AREA_MIN_PELOTA = 2
AREA_MAX_PELOTA = 90
LADO_MAX_PELOTA = 22
HUECO_MAX_INTERPOLAR = 6    # fotogramas sin pelota que se rellenan interpolando
ESTELA_PELOTA = 10          # nº de posiciones que se dibujan como estela

# --- Métricas ---
SUAVIZADO_POSICION = 7      # ventana (fotogramas) de la media móvil de posiciones
VELOCIDAD_MAX_MS = 9.0      # velocidades por encima se consideran ruido
