# Análisis de partidos de tenis con visión por computador clásica

Aplicación que, a partir de un vídeo grabado con **cámara fija** detrás de una pista de tenis,
detecta la pista, sigue a los dos jugadores y la pelota, y calcula la distancia recorrida y la
velocidad de cada jugador **en metros reales**.

Sólo se usan técnicas clásicas de OpenCV: **ningún modelo de IA, Machine Learning ni Deep Learning**.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

1. Copia tu vídeo en la carpeta `videos/`. En `videos/LEEME.md` está la lista de vídeos de prueba y de dónde se descargan.
2. Ejecuta, desde la carpeta del proyecto:

```bash
python main.py videos/partido_prueba.mp4
```

Opciones:

| Opción | Para qué sirve |
|---|---|
| `--manual` | Si la pista no se detecta bien: marcas con el ratón las 4 esquinas de la pista de dobles (sup-izq, sup-der, inf-der, inf-izq). Se guardan y se reutilizan. |
| `--max-frames N` | Procesa sólo los N primeros fotogramas (para pruebas rápidas). |
| `--sin-pelota` | Desactiva el seguimiento de la pelota. |

## Resultados (`salida/<nombre_del_video>/`)

| Fichero | Contenido |
|---|---|
| `<nombre>_analizado.mp4` | Vídeo con pista, jugadores, estela de la pelota, minimapa y estadísticas |
| `pista_detectada.jpg` | Comprobación de la detección de la pista (revísala siempre) |
| `fondo.jpg`, `mascara_lineas.png` | Imágenes intermedias (útiles para la memoria) |
| `mapa_calor.png` | Zonas por las que se ha movido cada jugador |
| `posiciones.csv` | Posición de cada jugador (m) y de la pelota (px) en cada fotograma |
| `resumen.json` | Distancia total, velocidad media y máxima de cada jugador |

## Estructura del código

| Fichero | Técnicas |
|---|---|
| `pista.py` | Top-hat morfológico (erosión, dilatación y resta), filtrado HSV con `inRange`, transformada de Hough probabilística, agrupación de rectas, búsqueda de hipótesis puntuada contra el modelo de pista, homografía con RANSAC |
| `jugadores.py` | Fondo por mediana temporal, resta con el fondo, suavizado por media, apertura y cierre con erosión y dilatación, contornos, homografía inversa del punto de los pies, asignación por mitades con continuidad temporal |
| `pelota.py` | Diferencia de tres fotogramas, umbral de brillo, contornos, filtro por tamaño, confirmación por tripletes de velocidad constante, interpolación |
| `metricas.py` | Suavizado, distancia y velocidad en metros, minimapa cenital, mapas de calor |
| `config.py` | Todos los parámetros ajustables (umbrales, tamaños...) |
| `main.py` | Programa principal |

En `FUNCIONES.md` está la documentación de todas las funciones utilizadas y el debate de clase en que se vio cada una.

## Requisitos del vídeo

- Cámara **fija** (trípode) durante todo el clip, con la pista completa a la vista, desde detrás de una línea de fondo y algo elevada.
- Si la cámara se mueve o cambia de plano, el fondo por mediana y la homografía dejan de valer.
- A más resolución y fotogramas por segundo, mejor se detecta la pelota (recomendado 1080p a 60 fps).

## Limitaciones conocidas

- La pelota se localiza en la imagen, pero **no se proyecta al minimapa**: la homografía sólo es
  válida para puntos sobre el suelo y la pelota casi siempre va en el aire.
- Cuando la pelota pasa por delante de un jugador se pierde unos fotogramas (se interpola).
- Si los dos jugadores se cruzan en la misma mitad (subida a la red) puede confundirse la identidad.
