# Documentación de las funciones utilizadas

El proyecto usa solo **OpenCV** y **NumPy**, sin ningún modelo de IA, Machine Learning ni Deep Learning.

Casi todas las funciones se han visto en los debates de la asignatura. Se indica en cuál.
Solo dos funciones de OpenCV no se vieron en clase: `cv2.HoughLinesP` y `cv2.findHomography`.
Las dos están en los tutoriales oficiales de OpenCV para Python (docs.opencv.org), como recomienda
el debate 39. Se explican con más detalle al final, porque son imprescindibles para el proyecto y
no tienen sustituto entre las funciones vistas.

## 1. Funciones vistas en clase

### Vídeo, imágenes y ventanas

| Función | Uso en el proyecto | Fichero | Debates |
|---|---|---|---|
| `cv2.VideoCapture` | Abrir el vídeo del partido y leerlo fotograma a fotograma (`read`) | main.py, jugadores.py | 64–67 |
| `cv2.VideoWriter`, `cv2.VideoWriter_fourcc` | Guardar el vídeo analizado (códec `mp4v`) | main.py | 65, 66 |
| `cv2.imwrite` | Guardar el fondo, la máscara de líneas, la pista detectada y el mapa de calor | main.py, metricas.py | 8, 65 |
| `cv2.imshow`, `cv2.waitKey`, `cv2.destroyAllWindows` | Ventana para marcar la pista a mano (modo `--manual`) | pista.py | casi todos |
| `cv2.namedWindow`, `cv2.setMouseCallback` | Recoger los 4 clics del ratón en las esquinas de la pista | pista.py | 17, 18–22 |
| `cv2.resize` | Reducir cada fotograma a 960 px de ancho para procesar más rápido | jugadores.py | 23 |

### Procesado de imagen

| Función | Uso en el proyecto | Fichero | Debates |
|---|---|---|---|
| `cv2.cvtColor` (`BGR2GRAY`, `BGR2HSV`) | Pasar a gris (diferencias, top-hat) y a HSV (brillo y saturación) | pista.py, main.py | 9, 28, 29 |
| `cv2.inRange` | Máscara de píxeles blancos: saturación baja y brillo alto en HSV | pista.py | 28, 79 |
| `cv2.threshold` (`THRESH_BINARY`) | Binarizar diferencias, top-hat y brillo | pista.py, jugadores.py, pelota.py | 31, 32, 44–48 |
| `cv2.blur` | Suavizado por media antes de comparar con el fondo, y suavizado del mapa de calor | jugadores.py, metricas.py | 34, 40 |
| `cv2.subtract` | Top-hat (original − apertura) y diferencia absoluta (ver abajo) | pista.py, jugadores.py | 25, 40, 54 |
| `cv2.add` | Suma de las dos restas para obtener la diferencia absoluta | jugadores.py | 26 |
| `cv2.bitwise_and` | Combinar máscaras (líneas ∧ blanco; movimiento ∧ brillo ∧ zona válida) | pista.py, pelota.py | 26, 28 |
| `cv2.erode`, `cv2.dilate` | Apertura, cierre y top-hat construidos a mano (ver abajo) | pista.py, jugadores.py, pelota.py | 35, 36, 79 |
| `np.ones` | Kernels de la morfología | pista.py, jugadores.py, pelota.py | 35, 36, 51 |

### Contornos

| Función | Uso en el proyecto | Fichero | Debates |
|---|---|---|---|
| `cv2.findContours` (`RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`) | Separar las manchas de movimiento: personas y candidatos a pelota | jugadores.py, pelota.py | 44–48, 50 |
| `cv2.contourArea` | Descartar manchas pequeñas que no pueden ser un jugador | jugadores.py | 46–48 |
| `cv2.boundingRect` | Rectángulo del jugador (y punto de sus pies) y tamaño de la pelota | jugadores.py, pelota.py | 50, 51, 54 |

### Dibujo

| Función | Uso en el proyecto | Fichero | Debates |
|---|---|---|---|
| `cv2.line` | Líneas de la pista proyectadas y estelas del minimapa | pista.py, metricas.py | 10, 19, 80 |
| `cv2.circle` | Pelota, estela de la pelota y jugadores en el minimapa | main.py, metricas.py | 14, 21, 79 |
| `cv2.rectangle` | Recuadros de los jugadores y marco del minimapa | main.py, metricas.py | 12, 13, 50 |
| `cv2.putText` | Nombres y estadísticas (distancia, velocidad) | main.py, metricas.py | 15, 16 |
| `cv2.addWeighted` | Minimapa semitransparente sobre el vídeo | metricas.py | 24 |

## 2. Técnicas construidas con funciones vistas en clase

Algunas operaciones tienen una función propia en OpenCV que no se vio en clase. En esos casos se
han programado combinando funciones que sí se vieron:

- **Diferencia absoluta |a − b|**: `cv2.subtract` pone a 0 los resultados negativos. Por eso
  `cv2.add(cv2.subtract(a, b), cv2.subtract(b, a))` da el valor absoluto: en cada píxel, una de las
  dos restas vale 0.
- **Apertura** = `dilate(erode(img))`. Elimina puntos sueltos de ruido.
- **Cierre** = `erode(dilate(img))`. Une brazos, piernas y raqueta al cuerpo del jugador.
- **Top-hat** = `subtract(img, apertura)`. La apertura borra las estructuras claras más finas que el
  kernel. Al restarla de la original solo quedan esas estructuras: las líneas de la pista.
- **Proyección de puntos con una homografía**: es una multiplicación de matrices en coordenadas
  homogéneas, `[x', y', w'] = H · [x, y, 1]`, y el resultado es `(x'/w', y'/w')`. Está hecha con NumPy
  en `pista.proyectar`.
- **Ajuste de una recta a un conjunto de puntos**: mínimos cuadrados con la descomposición SVD de
  NumPy (`np.linalg.svd`). La dirección de la recta es la de máxima dispersión de los puntos.
- **Zona válida de la pelota**: un píxel está dentro del cuadrilátero de la pista ampliada si
  queda al mismo lado de sus 4 lados. Se comprueba con el signo del producto vectorial en NumPy.
- **Centro y área de cada mancha**: se cuentan los píxeles blancos dentro de su `boundingRect`
  (`np.count_nonzero`) y se calcula la media de sus coordenadas.
- **Mapa de calor**: escala de color propia de amarillo (poco tiempo) a rojo (mucho tiempo).

## 3. Funciones de OpenCV no vistas en clase

### `cv2.HoughLinesP` — transformada de Hough probabilística

Documentación oficial: [Hough Line Transform (tutorial de OpenCV para Python)](https://docs.opencv.org/4.11.0/d6/d10/tutorial_py_houghlines.html)

```python
segmentos = cv2.HoughLinesP(mascara, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=ancho*0.06, maxLineGap=12)
```

**Qué hace.** Encuentra segmentos rectos en una imagen binaria. Cada píxel blanco "vota" por todas
las rectas que podrían pasar por él, expresadas como distancia al origen (ρ) y ángulo (θ). Las
rectas con muchos votos existen en la imagen. La versión probabilística (la P del nombre) solo usa
una muestra aleatoria de píxeles y devuelve directamente los extremos de cada segmento
`(x1, y1, x2, y2)`.

**Parámetros.**
- `rho`: resolución de la distancia, en píxeles.
- `theta`: resolución del ángulo (1°).
- `threshold`: votos mínimos para aceptar una recta.
- `minLineLength`: longitud mínima del segmento.
- `maxLineGap`: hueco máximo que se permite dentro de una misma línea, por ejemplo cuando un
  jugador la tapa en parte.

**Para qué se usa.** Para detectar las líneas de la pista en la máscara de líneas blancas. Los
segmentos se separan en horizontales (fondo y saque) y oblicuas (laterales), y se agrupan los que
están muy próximos. Sus intersecciones dan las esquinas de la pista.

**Por qué no hay alternativa entre lo visto en clase.** `findContours` devuelve contornos de
regiones, no rectas. Con él no se pueden obtener las ecuaciones de las líneas ni sus
intersecciones cuando las líneas se cortan o están parcialmente tapadas.

### `cv2.findHomography` — homografía entre dos planos

Documentación oficial: [Feature Matching + Homography to find Objects (tutorial de OpenCV para Python)](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html).
En el proyecto solo se usa `findHomography` con RANSAC, sin la parte de detección de características.

```python
H, _ = cv2.findHomography(puntos_pista_metros, puntos_imagen_px)                  # 4 puntos: exacta
H, _ = cv2.findHomography(puntos_pista_metros, puntos_imagen_px, cv2.RANSAC, 3.0) # muchos puntos: robusta
```

**Qué hace.** Calcula la matriz 3×3 `H` que transforma un plano en otro bajo perspectiva. Aquí
transforma el suelo de la pista (en metros, visto desde arriba) en la imagen de la cámara (en
píxeles).
- Con 4 parejas de puntos la solución es exacta.
- Con muchas parejas se usa **RANSAC** (*Random Sample Consensus*). RANSAC prueba muchas
  homografías calculadas con 4 parejas elegidas al azar y se queda con la que más parejas cumplen,
  así que ignora las erróneas. Es un método de estimación robusta, no de aprendizaje automático.

**Para qué se usa.**
1. **Comprobar hipótesis de pista**: para cada combinación de 2 líneas horizontales y 2 oblicuas
   se calcula `H` con sus 4 esquinas. Después se mide cuántas líneas del modelo oficial caen sobre
   píxeles blancos.
2. **Refinar** la mejor `H` con cientos de puntos de las líneas, usando RANSAC.
3. **Medir en metros**: con la inversa `H⁻¹`, el punto de los pies de cada jugador (centro del
   borde inferior de su rectángulo) se lleva a la pista real. Así la distancia recorrida y la
   velocidad se calculan en metros y no dependen de la perspectiva.

**Por qué no hay alternativa entre lo visto en clase.** Sin homografía no se puede pasar de
píxeles a metros. Un jugador al fondo de la imagen se ve mucho más pequeño que uno cercano, así que
la misma distancia en píxeles equivale a muchos más metros al fondo que cerca.

### Constantes y propiedades auxiliares

| Constante | Uso |
|---|---|
| `cv2.CAP_PROP_FPS`, `CAP_PROP_FRAME_COUNT`, `CAP_PROP_FRAME_WIDTH/HEIGHT` | Leer los fotogramas por segundo, el número de fotogramas y el tamaño del vídeo (`cap.get`) |
| `cv2.CAP_PROP_POS_FRAMES` | Saltar a un fotograma concreto (`cap.set`) para calcular el fondo |
| `cv2.INTER_AREA` | Interpolación de `resize` recomendada para reducir imágenes |
| `cv2.LINE_AA` | Dibujo con antialiasing (bordes suaves) |
| `cv2.FONT_HERSHEY_SIMPLEX` | Tipo de letra de `putText` (debates 15, 16) |
| `cv2.EVENT_LBUTTONDOWN` | Clic izquierdo del ratón en el modo manual |
