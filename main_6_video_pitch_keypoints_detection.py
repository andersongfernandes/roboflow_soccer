# -*- coding: utf-8 -*-
from tqdm import tqdm
import supervision as sv
import numpy as np
import os

# ------------------------------------------------------------------
# Providers do ONNX Runtime (sem colchetes e sem aspas extras)
# ------------------------------------------------------------------
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "DmlExecutionProvider,CPUExecutionProvider"

# ------------------------------------------------------------------
# Desliga modelos opcionais que exigem transformers/torch
# (mantém o ambiente leve como no seu exemplo)
# ------------------------------------------------------------------
for var in [
    "FLORENCE2_ENABLED","QWEN_2_5_ENABLED","CORE_MODEL_SAM_ENABLED","CORE_MODEL_SAM2_ENABLED",
    "CORE_MODEL_CLIP_ENABLED","CORE_MODEL_GAZE_ENABLED","SMOLVLM2_ENABLED","DEPTH_ESTIMATION_ENABLED",
    "MOONDREAM2_ENABLED","CORE_MODEL_TROCR_ENABLED","CORE_MODEL_GROUNDINGDINO_ENABLED",
    "CORE_MODEL_YOLO_WORLD_ENABLED","CORE_MODEL_PE_ENABLED"
]:
    os.environ[var] = "False"

from inference import get_model

# ------------------------------------------------------------------
# Model / paths
# ------------------------------------------------------------------
ROBOFLOW_API_KEY = "Q81j12ROpMGU5e7dBqkO"
PITCH_DETECTION_MODEL_ID = "football-field-detection-f07vi/15"
PITCH_DETECTION_MODEL = get_model(
    model_id=PITCH_DETECTION_MODEL_ID,
    api_key=ROBOFLOW_API_KEY
)

SOURCE_VIDEO_PATH = "video_data/match.mp4"
TARGET_VIDEO_PATH = "video_data/match_keypoints.mp4"
os.makedirs(os.path.dirname(TARGET_VIDEO_PATH), exist_ok=True)

# ------------------------------------------------------------------
# Annotators
# ------------------------------------------------------------------
# Cores vivas (paleta fixa) + raio maior para destacar bem os pontos
vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.from_hex("#FF1493"),  # rosa forte
    radius=8
)

# (Opcional) rotular os pontos por índice; útil para depuração
label_annotator = sv.LabelAnnotator(
    color=sv.Color.from_hex("#FFD700"),  # amarelo
    text_color=sv.Color.from_hex("#000000"),
    text_position=sv.Position.TOP_CENTER
)

# ------------------------------------------------------------------
# Vídeo I/O
# ------------------------------------------------------------------
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# ------------------------------------------------------------------
# Parâmetros de pós-processamento
# ------------------------------------------------------------------
CONF_THRESHOLD = 0.50   # mostra só pontos com confiança > 0.5
SHOW_LABELS = False     # mude para True se quiser ver #índice dos pontos

# ------------------------------------------------------------------
# Loop de processamento
# ------------------------------------------------------------------
with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        # Inferência
        result = PITCH_DETECTION_MODEL.infer(frame, confidence=0.30)[0]

        # Converte para estrutura de keypoints do Supervision
        key_points = sv.KeyPoints.from_inference(result)

        # Se o modelo não retornou nada, só grava o frame original
        if len(key_points) == 0 or key_points.xy is None or len(key_points.xy) == 0:
            video_sink.write_frame(frame)
            continue

        # Consideramos apenas o primeiro conjunto (índice 0) — ajuste se houver múltiplos
        kp_xy_all = key_points.xy[0]            # shape: (N, 2)
        kp_conf_all = key_points.confidence[0]  # shape: (N,)

        # Máscara por confiança
        mask = kp_conf_all > CONF_THRESHOLD

        # Se nada passou no filtro, grava frame sem anotações
        if not np.any(mask):
            video_sink.write_frame(frame)
            continue

        # Filtra e recria objeto KeyPoints com shape (1, M, 2)
        kp_xy = kp_xy_all[mask]
        kp_xy = kp_xy[np.newaxis, ...]
        filtered_keypoints = sv.KeyPoints(xy=kp_xy)

        # Desenha
        annotated = frame.copy()
        annotated = vertex_annotator.annotate(annotated, filtered_keypoints)

        # (Opcional) desenhar labels com o índice original do ponto
        if SHOW_LABELS:
            # índices originais (antes do filtro)
            orig_indices = np.nonzero(mask)[0].tolist()
            labels = [f"{i}" for i in orig_indices]
            annotated = label_annotator.annotate(annotated, filtered_keypoints, labels=labels)

        # Grava frame anotado
        video_sink.write_frame(annotated)

print(f"✅ Vídeo gerado em: {TARGET_VIDEO_PATH}")
