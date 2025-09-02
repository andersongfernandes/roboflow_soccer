# -*- coding: utf-8 -*-
"""
Gera imagens (PNG) da disposição tática a cada N segundos do vídeo.
Baseado no seu código de referência, mas sem exportar vídeo — apenas snapshots.
"""

# --- TOP OF FILE (primeiras linhas) ---
import os, multiprocessing, warnings

# Silencia o warning do loky definindo explicitamente os núcleos
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())
# Opcional: evita paralelismo via loky no Windows (reduz ruído de warnings)
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# Se usar onnxruntime com DirectML:
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[DmlExecutionProvider, CPUExecutionProvider]"

# Desliga modelos opcionais (os que dependeriam de torch/transformers extras)
for var in [
    "FLORENCE2_ENABLED","QWEN_2_5_ENABLED","CORE_MODEL_SAM_ENABLED","CORE_MODEL_SAM2_ENABLED",
    "CORE_MODEL_CLIP_ENABLED","CORE_MODEL_GAZE_ENABLED","SMOLVLM2_ENABLED","DEPTH_ESTIMATION_ENABLED",
    "MOONDREAM2_ENABLED","CORE_MODEL_TROCR_ENABLED","CORE_MODEL_GROUNDINGDINO_ENABLED",
    "CORE_MODEL_YOLO_WORLD_ENABLED","CORE_MODEL_PE_ENABLED"
]:
    os.environ[var] = "False"

# Opcional: suprime o UserWarning específico do loky
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context"
)

# ------------------- Imports -------------------
from tqdm import tqdm
import supervision as sv
import numpy as np
import cv2
from pathlib import Path

from sports.common.team import TeamClassifier
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

import torch
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked  # (mantido caso seu TeamClassifier use batching)

# ------------------- Configs -------------------
from inference import get_model

# Caminho do vídeo e saída
SOURCE_VIDEO_PATH = "video_data/match.mp4"
OUTPUT_DIR = "snapshots"
SNAP_EVERY_SECONDS = 5  # gere um snapshot a cada 5s (ajuste aqui)

# Modelos Roboflow (ajuste se necessário)
ROBOFLOW_API_KEY = "Q81j12ROpMGU5e7dBqkO"
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PITCH_DETECTION_MODEL_ID  = "football-field-detection-f07vi/15"

PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
PITCH_DETECTION_MODEL  = get_model(model_id=PITCH_DETECTION_MODEL_ID,  api_key=ROBOFLOW_API_KEY)

# IDs de classe do seu dataset
BALL_ID       = 0
GOALKEEPER_ID = 1
PLAYER_ID     = 2
REFEREE_ID    = 3  # se o modelo estiver treinado p/ árbitros

# Embeddings p/ TeamClassifier opcional (mantido do seu código)
SIGLIP_VISION_PATH = 'google/siglip-base-patch16-224'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_VISION_PATH).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_VISION_PATH)

BATCH_SIZE = 32

# ------------------- Utilitários -------------------
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_image(path: str | Path, img_rgb: np.ndarray) -> None:
    """
    Substitui sv.save_image: garante uint8 e converte RGB->BGR antes de salvar via cv2.imwrite.
    """
    if img_rgb is None:
        raise ValueError("Imagem vazia ao salvar.")

    img = img_rgb
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img

    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise IOError(f"Falha ao salvar imagem em {path}")

class ViewTransformer:
    """Homografia para projetar do frame para o pitch canônico."""
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)

    def transform_points(self, points: np.array) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)

def resolve_goalkeepers_team_id(players_detections: sv.Detections,
                                goalkeepers_detections: sv.Detections) -> np.ndarray:
    """Mesmo algoritmo do seu snippet — atribui goleiro ao time mais próximo do centróide."""
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

# ------------------- Pitch Config -------------------
CONFIG = SoccerPitchConfiguration()

# ------------------- Pipeline -------------------
def collect_sample_timestamps(video_path: str, every_seconds: int) -> list[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if frame_count > 0 else 0.0
    cap.release()

    if duration <= 0:
        # fallback: gere 1 min de timestamps se duração desconhecida
        duration = 60.0
    times = np.arange(0.0, duration + 0.001, every_seconds, dtype=float).tolist()
    # Garante pelo menos um frame (0s)
    if len(times) == 0 or times[0] != 0.0:
        times = [0.0] + times
    # Evita ultrapassar duração (tolerância de 0.25s)
    times = [t for t in times if t <= duration + 0.25]
    return times

def grab_frame_at(video_path: str, timestamp_s: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp_s * 1000.0))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

def detect_on_frame(frame: np.ndarray):
    """Roda o modelo de detecção e retorna Detections já com NMS (exceto bola que é tratada à parte)."""
    results = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(results)
    return detections

def fit_team_classifier_from_samples(samples_frames: list[np.ndarray]) -> TeamClassifier | None:
    """
    Coleta crops de jogadores nas amostras e faz fit do TeamClassifier.
    Evita percorrer o vídeo inteiro, usando apenas os timestamps escolhidos.
    """
    all_crops = []
    for frame in tqdm(samples_frames, desc='coletando crops p/ TeamClassifier'):
        det = detect_on_frame(frame)
        # apenas jogadores (PLAYER_ID)
        det_players = det[det.class_id == PLAYER_ID]
        det_players = det_players.with_nms(threshold=0.5, class_agnostic=True)
        if len(det_players) > 0:
            all_crops.extend([sv.crop_image(frame, xyxy) for xyxy in det_players.xyxy])

    if len(all_crops) == 0:
        return None

    # (Opcional) extrair embeddings SigLIP — seu TeamClassifier pode trabalhar só com crops,
    # então aqui mantemos o comportamento simples: apenas fit(crops)
    team_classifier = TeamClassifier(device=DEVICE)
    team_classifier.fit(all_crops)
    return team_classifier

def compute_pitch_homography(frame: np.ndarray) -> 'ViewTransformer | None':
    """Detecta pontos de referência do campo e retorna o transformador de vista (frame -> pitch)."""
    result = PITCH_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)
    if key_points.xy is None or len(key_points.xy) == 0:
        return None

    mask = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][mask]
    if frame_reference_points.shape[0] < 4:
        return None

    pitch_reference_points = np.array(CONFIG.vertices)[mask]

    try:
        vt = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)
        return vt
    except Exception:
        return None

def draw_snapshot_pitch(frame: np.ndarray,
                        detections: sv.Detections,
                        team_classifier: TeamClassifier | None,
                        vt: ViewTransformer | None) -> np.ndarray:
    """
    Constrói a imagem do pitch com bola, jogadores, goleiros e árbitros projetados.
    """
    # Separa por classes
    ball = detections[detections.class_id == BALL_ID]
    others = detections[detections.class_id != BALL_ID].with_nms(threshold=0.5, class_agnostic=True)

    players = others[others.class_id == PLAYER_ID]
    goalies = others[others.class_id == GOALKEEPER_ID]
    refs    = others[others.class_id == REFEREE_ID]

    # Predição de time para jogadores via TeamClassifier
    if team_classifier is not None and len(players) > 0:
        player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
        if len(player_crops) > 0:
            players.class_id = team_classifier.predict(player_crops)
        else:
            players.class_id = np.zeros(len(players), dtype=int)
    else:
        if len(players) > 0:
            players.class_id = np.zeros(len(players), dtype=int)

    # Predição de time para goleiros (centróide nearest-team)
    if len(goalies) > 0:
        goalies.class_id = resolve_goalkeepers_team_id(players, goalies)

    # Posição no frame (centro da base da bbox)
    frame_ball_xy     = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(ball) > 0 else np.zeros((0, 2))
    frame_players_xy  = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(players) > 0 else np.zeros((0, 2))
    frame_refs_xy     = refs.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(refs) > 0 else np.zeros((0, 2))

    # Projeção para o pitch
    if vt is not None:
        pitch_ball_xy    = vt.transform_points(frame_ball_xy)
        pitch_players_xy = vt.transform_points(frame_players_xy)
        pitch_refs_xy    = vt.transform_points(frame_refs_xy)
    else:
        pitch_ball_xy    = np.zeros((0, 2))
        pitch_players_xy = np.zeros((0, 2))
        pitch_refs_xy    = np.zeros((0, 2))

    # Desenho do pitch (RGB)
    pitch_img = draw_pitch(config=CONFIG)

    # Bola
    if len(pitch_ball_xy) > 0:
        pitch_img = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=pitch_img
        )

    # Jogadores time 0 e time 1
    if len(pitch_players_xy) > 0:
        mask_t0 = (players.class_id == 0)
        mask_t1 = (players.class_id == 1)
        if np.any(mask_t0):
            pitch_img = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_xy[mask_t0],
                face_color=sv.Color.from_hex("#00BFFF"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch_img
            )
        if np.any(mask_t1):
            pitch_img = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_xy[mask_t1],
                face_color=sv.Color.from_hex("#FF1493"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch_img
            )

    # Árbitros (amarelo)
    if len(pitch_refs_xy) > 0:
        pitch_img = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_refs_xy,
            face_color=sv.Color.from_hex("#FFD700"),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=pitch_img
        )

    return pitch_img

def main():
    ensure_dir(OUTPUT_DIR)

    # 1) Coleta timestamps
    timestamps = collect_sample_timestamps(SOURCE_VIDEO_PATH, SNAP_EVERY_SECONDS)
    if len(timestamps) == 0:
        raise RuntimeError("Nenhum timestamp gerado. Verifique o vídeo/configurações.")

    # 2) Carrega frames das amostras (uma única vez) p/ fit do TeamClassifier
    sample_frames_for_fit = []
    for t in timestamps:
        f = grab_frame_at(SOURCE_VIDEO_PATH, t)
        if f is not None:
            sample_frames_for_fit.append(f)
    if len(sample_frames_for_fit) == 0:
        raise RuntimeError("Não foi possível ler frames do vídeo para os timestamps escolhidos.")

    # 3) Fit do TeamClassifier (somente com amostras)
    team_classifier = fit_team_classifier_from_samples(sample_frames_for_fit)
    if team_classifier is None:
        print("[AVISO] Não foi possível treinar o TeamClassifier com as amostras; usando fallback (time 0).")

    # 4) Processa cada timestamp e salva o snapshot do pitch
    for t in tqdm(timestamps, desc="gerando snapshots"):
        frame = grab_frame_at(SOURCE_VIDEO_PATH, t)
        if frame is None:
            print(f"[aviso] não consegui ler o frame em t={t:.2f}s; pulando...")
            continue

        detections = detect_on_frame(frame)
        vt = compute_pitch_homography(frame)  # tenta homografia deste frame

        pitch_img = draw_snapshot_pitch(frame, detections, team_classifier, vt)

        # Salva arquivo (usa helper baseado em OpenCV)
        fname = f"snapshot_{int(round(t)):03d}s.png"
        out_path = Path(OUTPUT_DIR) / fname
        save_image(out_path, pitch_img)

    print(f"Concluído! Imagens salvas em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
