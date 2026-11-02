"""
Análisis automático de partidos de tenis con visión por computador clásica.

Uso:
    python main.py videos/mi_partido.mp4
    python main.py videos/mi_partido.mp4 --manual        # marcar la pista a mano
    python main.py videos/mi_partido.mp4 --max-frames 300

Resultados en la carpeta salida/<nombre_del_video>/
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

import config as C
import metricas
import pelota
import pista
from jugadores import SeguidorJugadores, calcular_fondo, mascara_primer_plano, redimensionar


def escalar_homografia(H, s):
    """H trabaja a resolución reducida; la pasamos a la resolución original."""
    S_inv = np.diag([1 / s, 1 / s, 1.0])
    return S_inv @ H


def main():
    ap = argparse.ArgumentParser(description="Análisis de partidos de tenis (OpenCV clásico)")
    ap.add_argument("video", help="ruta al vídeo (p. ej. videos/partido.mp4)")
    ap.add_argument("--manual", action="store_true", help="marcar las 4 esquinas de la pista a mano")
    ap.add_argument("--max-frames", type=int, default=0, help="procesar sólo los N primeros fotogramas")
    ap.add_argument("--sin-pelota", action="store_true", help="no buscar la pelota")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        raise SystemExit(f"No encuentro el vídeo '{args.video}'. ¿Lo has puesto en la carpeta videos/?")

    nombre = os.path.splitext(os.path.basename(args.video))[0]
    carpeta = os.path.join("salida", nombre)
    os.makedirs(carpeta, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, Hh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    ancho = min(C.ANCHO_TRABAJO, W)
    s = ancho / W
    print(f"[video] {W}x{Hh} a {fps:.1f} fps, {total} fotogramas")

    # ------------------------------------------------------------------
    # 1. Fondo y pista
    # ------------------------------------------------------------------
    t0 = time.time()
    fondo = calcular_fondo(args.video, ancho)
    cv2.imwrite(os.path.join(carpeta, "fondo.jpg"), fondo)

    if args.manual:
        H = pista.homografia_manual(fondo, os.path.join(carpeta, "esquinas.json"))
        calidad = 1.0
    else:
        H, calidad, mascara = pista.buscar_homografia(fondo)
        cv2.imwrite(os.path.join(carpeta, "mascara_lineas.png"), mascara)
        if H is None:
            raise SystemExit("No se ha encontrado la pista. Prueba con --manual")
    vista = pista.dibujar_pista(fondo.copy(), H, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(carpeta, "pista_detectada.jpg"), vista)
    print(f"[pista] coincidencia con el modelo: {calidad * 100:.0f}% "
          f"(revisa salida/{nombre}/pista_detectada.jpg)")
    if calidad < 0.6:
        print("[pista] AVISO: coincidencia baja; si las líneas no encajan usa --manual")

    # ------------------------------------------------------------------
    # 2. Primera pasada: detección de jugadores y candidatos de pelota
    # ------------------------------------------------------------------
    seguidor = SeguidorJugadores(H)
    zona = pelota.zona_valida(H, fondo.shape[:2])
    cajas = []                                 # por fotograma: [bbox o None] x2
    pos_m = np.full((2, total, 2), np.nan)     # posiciones en metros
    cands = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    buffer = []                                # (gris, hsv) de t-1, t, t+1

    def procesar_pelota(t):
        if args.sin_pelota or len(buffer) < 3:
            cands.append([])
            return
        (g0, _), (g1, hsv1), (g2, _) = buffer
        cajas_t = [c[0] for c in cajas[t] if c is not None]
        cands.append(pelota.candidatos(g0, g1, g2, hsv1, zona, cajas_t))

    for t in range(total):
        ok, frame = cap.read()
        if not ok:
            total = t
            pos_m = pos_m[:, :t]
            break
        img = redimensionar(frame, ancho)
        jug = seguidor.actualizar(mascara_primer_plano(img, fondo))
        cajas.append(jug)
        for j in range(2):
            if jug[j] is not None:
                pos_m[j, t] = jug[j][1]

        buffer.append((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2HSV)))
        if len(buffer) > 3:
            buffer.pop(0)
        if t == 0:
            continue
        procesar_pelota(t - 1)                 # la pelota en t-1 necesita el fotograma t
        if t % 50 == 0:
            print(f"[detección] {t}/{total}")
    # cands[t] corresponde al fotograma t; el último no tiene fotograma siguiente
    cands = cands[:total]
    while len(cands) < total:
        cands.append([])
    cap.release()

    tray_pelota = pelota.construir_trayectoria(cands) if not args.sin_pelota else np.full((total, 2), np.nan)
    detectados = [int((~np.isnan(pos_m[j, :, 0])).sum()) for j in range(2)]
    print(f"[jugadores] detectados en {detectados[0]}/{total} y {detectados[1]}/{total} fotogramas")
    print(f"[pelota] localizada en {int((~np.isnan(tray_pelota[:, 0])).sum())}/{total} fotogramas")

    # ------------------------------------------------------------------
    # 3. Métricas
    # ------------------------------------------------------------------
    pos_suave = [metricas.rellenar_y_suavizar(pos_m[j]) for j in range(2)]
    dist, vel = zip(*[metricas.calcular_metricas(p, fps) for p in pos_suave])
    # si un jugador no se ha detectado nunca, sus métricas son 0 (y no NaN)
    dist = [np.nan_to_num(d) for d in dist]
    vel = [np.nan_to_num(v) for v in vel]

    # ------------------------------------------------------------------
    # 4. Segunda pasada: vídeo final
    # ------------------------------------------------------------------
    H_full = escalar_homografia(H, s)
    cap = cv2.VideoCapture(args.video)
    ruta_video = os.path.join(carpeta, f"{nombre}_analizado.mp4")
    salida = cv2.VideoWriter(ruta_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hh))
    minimapa = metricas.Minimapa(alto_px=Hh * 0.42)
    escala_ui = Hh / 1080

    for t in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        pista.dibujar_pista(frame, H_full, (0, 255, 255), max(1, int(2 * escala_ui)))

        for j in range(2):
            if cajas[t][j] is not None:
                x0, y0, x1, y1 = [int(v / s) for v in cajas[t][j][0]]
                cv2.rectangle(frame, (x0, y0), (x1, y1), metricas.COLORES[j], max(2, int(3 * escala_ui)))
                cv2.putText(frame, metricas.NOMBRES[j], (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7 * escala_ui, metricas.COLORES[j], 2, cv2.LINE_AA)

        # estela de la pelota
        desde = max(0, t - C.ESTELA_PELOTA)
        pts = [tray_pelota[k] / s for k in range(desde, t + 1) if not np.isnan(tray_pelota[k, 0])]
        for k, p in enumerate(pts):
            radio = max(2, int((3 + 5 * (k + 1) / len(pts)) * escala_ui))
            cv2.circle(frame, (int(p[0]), int(p[1])), radio, (0, 255, 255), -1, cv2.LINE_AA)
        if not np.isnan(tray_pelota[t, 0]):
            p = tray_pelota[t] / s
            cv2.circle(frame, (int(p[0]), int(p[1])), int(12 * escala_ui), (0, 0, 255), 2, cv2.LINE_AA)

        actuales = [pos_suave[j][t] for j in range(2)]
        estelas = [pos_suave[j][max(0, t - int(fps * 3)):t + 1] for j in range(2)]
        minimapa.dibujar(frame, actuales, estelas, W - minimapa.w - int(30 * escala_ui), int(30 * escala_ui))
        metricas.dibujar_marcador(frame, [dist[0][t], dist[1][t]], [vel[0][t], vel[1][t]],
                                  int(30 * escala_ui), int(30 * escala_ui), escala_ui)
        salida.write(frame)
        if t % 50 == 0:
            print(f"[vídeo] {t}/{total}")
    salida.release()
    cap.release()

    # ------------------------------------------------------------------
    # 5. Ficheros de resultados
    # ------------------------------------------------------------------
    metricas.mapa_calor(pos_suave, os.path.join(carpeta, "mapa_calor.png"))
    with open(os.path.join(carpeta, "posiciones.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fotograma", "segundo", "fondo_x_m", "fondo_y_m", "cerca_x_m", "cerca_y_m",
                    "pelota_x_px", "pelota_y_px"])
        for t in range(total):
            fila = [t, round(t / fps, 3)]
            for j in range(2):
                fila += [round(float(v), 3) for v in pos_suave[j][t]]
            fila += ["" if np.isnan(v) else round(float(v / s), 1) for v in tray_pelota[t]]
            w.writerow(fila)
    resumen = {
        "video": args.video,
        "duracion_s": round(total / fps, 2),
        "coincidencia_pista": round(float(calidad), 3),
        "jugadores": {
            metricas.NOMBRES[j]: {
                "distancia_m": round(float(dist[j][-1]), 1) if total else 0,
                "velocidad_media_kmh": round(float(np.mean(vel[j]) * 3.6), 1) if total else 0,
                "velocidad_max_kmh": round(float(np.max(vel[j]) * 3.6), 1) if total else 0,
                "fotogramas_detectado": detectados[j],
            } for j in range(2)
        },
        "fotogramas_con_pelota": int((~np.isnan(tray_pelota[:, 0])).sum()),
    }
    with open(os.path.join(carpeta, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(json.dumps(resumen["jugadores"], indent=2, ensure_ascii=False))
    print(f"[fin] {time.time() - t0:.1f} s. Resultados en {carpeta}/")


if __name__ == "__main__":
    main()
