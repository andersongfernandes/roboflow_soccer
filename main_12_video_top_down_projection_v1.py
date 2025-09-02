# -*- coding: utf-8 -*-
"""
Gera UM vídeo final (MP4) com a movimentação no pitch canônico:
- Time A, Time B, Bola e Árbitros (amarelo).
- Eficiência: detecção esparsa + tracking, homografia esparsa, bootstrap HSV leve.

Requisitos (principais):
- opencv-python, numpy, supervision, scikit-learn (apenas para KMeans leve),
- transformers/torch NÃO são usados para time (apenas detector/roboflow),
- seu 'inference.get_model' configurado para os modelos Roboflow.

Atenção:
- IDs de classe (BALL_ID/GOALKEEPER_ID/PLAYER_ID/REFEREE_ID) seguem seu dataset.
"""

# ---------------- Boot e ambiente ----------------
import os, multiprocessing, warnings
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[DmlExecutionProvider, CPUExecutionProvider]"
for var in [
    "FLORENCE2_ENABLED","QWEN_2_5_ENABLED","CORE_MODEL_SAM_ENABLED","CORE_MODEL_SAM2_ENABLED",
    "CORE_MODEL_CLIP_ENABLED","CORE_MODEL_GAZE_ENABLED","SMOLVLM2_ENABLED","DEPTH_ESTIMATION_ENABLED",
    "MOONDREAM2_ENABLED","CORE_MODEL_TROCR_ENABLED","CORE_MODEL_GROUNDINGDINO_ENABLED",
    "CORE_MODEL_YOLO_WORLD_ENABLED","CORE_MODEL_PE_ENABLED"
]:
    os.environ[var] = "False"
warnings.filterwarnings("ignore", category=UserWarning, module=r"joblib\.externals\.loky\.backend\.context")

# ---------------- Imports ----------------
import cv2
import numpy as np
from pathlib import Path
from collections import deque

import supervision as sv
from sklearn.cluster import KMeans  # leve, apenas para bootstrap HSV

from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.team import TeamClassifier  # não usado para treino; mantido se quiser alternar no futuro

# ---------------- Modelos Roboflow ----------------
from inference import get_model

# ---------------- Configs ----------------
# IO
SOURCE_VIDEO_PATH = "video_data/match.mp4"
OUTPUT_VIDEO_PATH = "output/minimapa.movimentacao.mp4"

# Roboflow
ROBOFLOW_API_KEY = "Q81j12ROpMGU5e7dBqkO"
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PITCH_DETECTION_MODEL_ID  = "football-field-detection-f07vi/15"

# Classes do dataset
BALL_ID       = 0
GOALKEEPER_ID = 1
PLAYER_ID     = 2
REFEREE_ID    = 3

# Pipeline
DETECT_EVERY        = 3    # detectar a cada N frames (tracking nos intermediários)
HOMOGRAPHY_EVERY    = 20   # recalcular homografia a cada N frames
BOOTSTRAP_SECONDS   = 2.0  # segundos iniciais para aprender HSV dos times
CONF_DET            = 0.30 # confiança mínima da detecção
NMS_IOU             = 0.50 # NMS class-agnostic
MAX_QUEUE_TRACK     = 32   # histórico curto (opcional p/ depuração)

# Visual
DOT_R_PLAYER = 14
DOT_R_BALL   = 9
COLOR_TEAM_A = sv.Color.from_hex("#00BFFF")  # ciano
COLOR_TEAM_B = sv.Color.from_hex("#FF1493")  # rosa
COLOR_REF    = sv.Color.from_hex("#FFD700")  # amarelo
COLOR_BALL   = sv.Color.WHITE

# Saída do vídeo (canvas do pitch)
PITCH_SCALE = 1.0   # 1.0 = resolução da configuração padrão; use 1.5/2.0 se quiser maior
OUTPUT_FPS_OVERRIDE = None  # defina um número p/ forçar FPS; None = usa FPS do vídeo de entrada

# ---------------- Utilitários ----------------
def ensure_parent_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def central_hsv_patch(img_bgr: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """
    Retorna os HSVs de um patch central da bbox para reduzir contaminação pelo gramado.
    frac = fração do lado (0.5 => 50% central).
    """
    h, w = img_bgr.shape[:2]
    cw, ch = int(w*frac), int(h*frac)
    x0 = (w - cw)//2
    y0 = (h - ch)//2
    patch = img_bgr[y0:y0+ch, x0:x0+cw]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3)

def median_hsv_of_crop(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = central_hsv_patch(crop_bgr, frac=0.5)
    med = np.median(hsv, axis=0)
    return med  # shape (3,)

def resolve_goalkeepers_team_id(players_detections: sv.Detections,
                                goalkeepers_detections: sv.Detections) -> np.ndarray:
    """Atribui goleiro ao time cujo centróide está mais próximo (barato e eficaz)."""
    if len(goalkeepers_detections) == 0:
        return np.array([], dtype=int)
    if len(players_detections) == 0:
        return np.zeros(len(goalkeepers_detections), dtype=int)

    has_team0 = np.any(players_detections.class_id == 0)
    has_team1 = np.any(players_detections.class_id == 1)

    goalkeepers_xy = goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    if not has_team0 and not has_team1:
        return np.zeros(len(goalkeepers_detections), dtype=int)
    if not has_team0:
        return np.ones(len(goalkeepers_detections), dtype=int)
    if not has_team1:
        return np.zeros(len(goalkeepers_detections), dtype=int)

    team_0_centroid = players_xy[players_detections.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_detections.class_id == 1].mean(axis=0)

    goalkeepers_team_ids = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_ids.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_ids, dtype=int)

class ViewTransformer:
    """Homografia frame -> pitch canônico."""
    def __init__(self, source: np.ndarray, target: np.ndarray):
        # source/target: Nx2
        self.m, _ = cv2.findHomography(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        pts = cv2.perspectiveTransform(pts, self.m)
        return pts.reshape(-1, 2).astype(np.float32)

# ---------------- Pitch e projeção ----------------
CONFIG = SoccerPitchConfiguration()
PITCH_BASE = draw_pitch(config=CONFIG)  # pitch base canônico (pré-renderizado)

def compute_pitch_homography(frame_bgr: np.ndarray, model) -> ViewTransformer | None:
    """Detecta keypoints do campo e ajusta homografia; retorna None se não houver pontos suficientes."""
    result = model.infer(frame_bgr, confidence=0.30)[0]
    key_points = sv.KeyPoints.from_inference(result)
    if key_points.xy is None or len(key_points.xy) == 0:
        return None
    mask = key_points.confidence[0] > 0.5
    frame_pts = key_points.xy[0][mask]
    if frame_pts.shape[0] < 4:
        return None
    pitch_pts = np.array(CONFIG.vertices)[mask]
    try:
        return ViewTransformer(frame_pts, pitch_pts)
    except Exception:
        return None

# ---------------- HSV Bootstrap ----------------
def bootstrap_team_hsv_centroids(capture: cv2.VideoCapture,
                                 seconds: float,
                                 detector,
                                 fps: float,
                                 conf_det: float,
                                 nms_iou: float,
                                 max_samples: int = 400) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Coleta HSV médios (mediana) de jogadores nos primeiros 'seconds' e faz KMeans(k=2).
    Retorna (centroidA, centroidB) em HSV, ou None se não for possível.
    """
    total_frames = int(seconds * fps)
    if total_frames <= 0:
        total_frames = int(2 * fps)

    hsv_samples = []
    curr_pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    end_pos  = min(curr_pos + total_frames, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

    for _ in range(curr_pos, end_pos):
        ok, frame = capture.read()
        if not ok:
            break
        result = detector.infer(frame, confidence=conf_det)[0]
        det = sv.Detections.from_inference(result)
        det = det.with_nms(threshold=nms_iou, class_agnostic=True)
        det_players = det[det.class_id == PLAYER_ID]
        if len(det_players) == 0:
            continue
        # coleta amostras de HSV medianos do patch central
        for xyxy in det_players.xyxy:
            crop = sv.crop_image(frame, xyxy)
            med_hsv = median_hsv_of_crop(crop)
            hsv_samples.append(med_hsv)
            if len(hsv_samples) >= max_samples:
                break
        if len(hsv_samples) >= max_samples:
            break

    hsv_samples = np.array(hsv_samples, dtype=np.float32)  # [N,3]
    if hsv_samples.shape[0] < 10:
        return None

    # KMeans k=2 para dois times
    kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42)
    kmeans.fit(hsv_samples)
    c0, c1 = kmeans.cluster_centers_  # (3,), (3,)
    return (c0, c1)

def assign_team_by_hsv(crop_bgr: np.ndarray, centroidA: np.ndarray, centroidB: np.ndarray) -> int:
    hsv_med = median_hsv_of_crop(crop_bgr)
    dA = np.linalg.norm(hsv_med - centroidA)
    dB = np.linalg.norm(hsv_med - centroidB)
    return 0 if dA <= dB else 1

# ---------------- Pipeline principal ----------------
def main():
    ensure_parent_dir(OUTPUT_VIDEO_PATH)

    # Modelos
    player_detector = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    pitch_detector  = get_model(model_id=PITCH_DETECTION_MODEL_ID,  api_key=ROBOFLOW_API_KEY)

    # Vídeo de entrada
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {SOURCE_VIDEO_PATH}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = OUTPUT_FPS_OVERRIDE if OUTPUT_FPS_OVERRIDE else in_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Canvas de saída: usamos o pitch canônico como base; pegamos sua resolução atual
    pitch_img0 = PITCH_BASE.copy()
    H, W = pitch_img0.shape[:2]
    if PITCH_SCALE != 1.0:
        W = int(W * PITCH_SCALE)
        H = int(H * PITCH_SCALE)
        pitch_img0 = cv2.resize(pitch_img0, (W, H), interpolation=cv2.INTER_LINEAR)

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, out_fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Não foi possível abrir o writer para: {OUTPUT_VIDEO_PATH}")

    # Bootstrap HSV dos times (rápido e leve)
    print("[INFO] Bootstrap HSV dos times...")
    start_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    boot = bootstrap_team_hsv_centroids(
        capture=cap,
        seconds=BOOTSTRAP_SECONDS,
        detector=player_detector,
        fps=in_fps,
        conf_det=CONF_DET,
        nms_iou=NMS_IOU,
        max_samples=400
    )
    # Após bootstrap, cap está adiante; reposiciona para o início real do processamento
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

    if boot is None:
        print("[AVISO] Bootstrap HSV insuficiente. Usando centroids default (tons distintos).")
        centroidA = np.array([100, 150, 150], dtype=np.float32)  # H,S,V (aprox ciano)
        centroidB = np.array([320 % 180, 150, 150], dtype=np.float32)  # wrap em H (rosa)
    else:
        centroidA, centroidB = boot

    # Tracking (ByteTrack via supervision)
    tracker = sv.ByteTrack()
    tracker.reset()

    # Homografia
    vt: ViewTransformer | None = None
    last_hg_frame = -999

    # Buffer (opcional p/ debug)
    last_positions = deque(maxlen=MAX_QUEUE_TRACK)

    # Loop principal
    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        need_detect = (f_idx % DETECT_EVERY == 0)

        if need_detect:
            # Detecção
            infer = player_detector.infer(frame, confidence=CONF_DET)[0]
            det   = sv.Detections.from_inference(infer)
            # Bola separada? (opcional: aumentar bbox da bola levemente p/ vis)
            ball = det[det.class_id == BALL_ID]
            # Restante com NMS e tracking
            non_ball = det[det.class_id != BALL_ID].with_nms(threshold=NMS_IOU, class_agnostic=True)
            tracked = tracker.update_with_detections(non_ball)

        else:
            # Sem nova detecção: reutiliza última e deixa o tracker propagar
            # (O supervision ByteTrack precisa da chamada update_with_detections mesmo assim.
            #  Uma estratégia simples é: sem detections novas -> passar det vazio e manter estado)
            tracked = tracker.update_with_detections(sv.Detections.empty())
            # Não temos 'ball' se não detectamos neste frame; manter “última bola conhecida”
            # será feito via buffer 'last_positions' logo abaixo.
            ball = sv.Detections.empty()

        # Recalcular homografia esparsamente
        if vt is None or (f_idx - last_hg_frame) >= HOMOGRAPHY_EVERY:
            vt_try = compute_pitch_homography(frame, pitch_detector)
            if vt_try is not None:
                vt = vt_try
                last_hg_frame = f_idx

        # Separar classes
        players  = tracked[tracked.class_id == PLAYER_ID]
        goalies  = tracked[tracked.class_id == GOALKEEPER_ID]
        referees = tracked[tracked.class_id == REFEREE_ID]

        # Atribuição de times por HSV (rápida)
        if len(players) > 0:
            # crops dos jogadores do frame atual (somente quando houve detecção)
            # se este frame não detectou (apenas tracking), players.xyxy virá do tracker; ainda dá para cropá-los
            crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
            labels = []
            for crop in crops:
                try:
                    t_id = assign_team_by_hsv(crop, centroidA, centroidB)
                except Exception:
                    t_id = 0
                labels.append(t_id)
            players.class_id = np.array(labels, dtype=int)
        else:
            # nenhum jogador; segue
            pass

        # Goleiros -> time via centróides
        if len(goalies) > 0:
            goalies.class_id = resolve_goalkeepers_team_id(players, goalies)

        # Coordenadas de ancoragem no frame
        frame_ball_xy    = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(ball) > 0 else np.zeros((0,2))
        frame_players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(players) > 0 else np.zeros((0,2))
        frame_refs_xy    = referees.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(referees) > 0 else np.zeros((0,2))

        # Projeção para pitch (se sem homografia, não plota)
        if vt is not None:
            pitch_ball_xy    = vt.transform_points(frame_ball_xy)
            pitch_players_xy = vt.transform_points(frame_players_xy)
            pitch_refs_xy    = vt.transform_points(frame_refs_xy)
        else:
            pitch_ball_xy    = np.zeros((0,2))
            pitch_players_xy = np.zeros((0,2))
            pitch_refs_xy    = np.zeros((0,2))

        # Desenho no pitch base (cópia por frame)
        canvas = pitch_img0.copy()

        # Bola
        if len(pitch_ball_xy) > 0:
            canvas = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_ball_xy,
                face_color=COLOR_BALL,
                edge_color=sv.Color.BLACK,
                radius=DOT_R_BALL,
                pitch=canvas
            )

        # Jogadores A/B
        if len(pitch_players_xy) > 0:
            maskA = (players.class_id == 0)
            maskB = (players.class_id == 1)
            if np.any(maskA):
                canvas = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[maskA],
                    face_color=COLOR_TEAM_A,
                    edge_color=sv.Color.BLACK,
                    radius=DOT_R_PLAYER,
                    pitch=canvas
                )
            if np.any(maskB):
                canvas = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[maskB],
                    face_color=COLOR_TEAM_B,
                    edge_color=sv.Color.BLACK,
                    radius=DOT_R_PLAYER,
                    pitch=canvas
                )

        # Árbitros
        if len(pitch_refs_xy) > 0:
            canvas = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_refs_xy,
                face_color=COLOR_REF,
                edge_color=sv.Color.BLACK,
                radius=DOT_R_PLAYER,
                pitch=canvas
            )

        # Redimensiona se necessário (já ajustado na criação de pitch_img0)
        out_frame = canvas

        # Escreve vídeo
        writer.write(out_frame)

        # (Opcional) guarda último estado para depuração
        last_positions.append({
            "f": f_idx,
            "n_players": len(players),
            "n_refs": len(referees),
            "n_goalies": len(goalies),
            "n_ball": len(ball)
        })

        f_idx += 1

    # Flush
    writer.release()
    cap.release()
    print(f"[OK] Vídeo gerado em: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()
