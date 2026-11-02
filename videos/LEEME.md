# Carpeta de vídeos

Los vídeos se guardan aquí, pero no forman parte del código. Antes de ejecutar el programa,
copia en esta carpeta los vídeos que quieras analizar.

Vídeos de prueba que usamos:

| Archivo | Partido | Pista | Resultado |
|---|---|---|---|
| `partido_prueba.mp4` | Djokovic – Sonego | Dura | Funciona |
| `video_input2.mp4` | Djokovic – Sonego (15 s, 60 fps) | Dura | Funciona |
| `video_input3.mp4` | Nadal – Dimitrov | Dura | Funciona |
| `video_input4.mp4` | Monfils – Paire | Dura | Funciona |
| `video_input6.mp4` | Shapovalov – Tiafoe | Dura | Funciona |
| `video_input8.mp4` | Tsitsipas – Rublev | Tierra | Necesita `--manual` |
| `video_input5.mp4` | Djokovic – Del Potro | Tierra | Falla: la cámara se mueve |

Origen:
- `partido_prueba.mp4`: repositorio público [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis) (`input_videos/input_video.mp4`).
- `video_input*.mp4`: repositorio público [ArtLabss/tennis-tracking](https://github.com/ArtLabss/tennis-tracking) (carpeta `VideoInput`).

Uso:

```
py main.py videos\partido_prueba.mp4
```
