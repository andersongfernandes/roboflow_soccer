from tqdm import tqdm
import supervision as sv
import numpy as np
import os

# Providers do ONNX Runtime (sem aspas extras nos nomes dos providers)
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "DmlExecutionProvider,CPUExecutionProvider"

# Desliga modelos opcionais que exigem transformers/torch
for var in [
    "FLORENCE2_ENABLED","QWEN_2_5_ENABLED","CORE_MODEL_SAM_ENABLED","CORE_MODEL_SAM2_ENABLED",
    "CORE_MODEL_CLIP_ENABLED","CORE_MODEL_GAZE_ENABLED","SMOLVLM2_ENABLED","DEPTH_ESTIMATION_ENABLED",
    "MOONDREAM2_ENABLED","CORE_MODEL_TROCR_ENABLED","CORE_MODEL_GROUNDINGDINO_ENABLED",
    "CORE_MODEL_YOLO_WORLD_ENABLED","CORE_MODEL_PE_ENABLED"
]:
    os.environ[var] = "False"

from inference import get_model

ROBOFLOW_API_KEY = "Q81j12ROpMGU5e7dBqkO"
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PLAYER_DETECTION_MODEL = get_model(
    model_id=PLAYER_DETECTION_MODEL_ID,
    api_key=ROBOFLOW_API_KEY
)

SOURCE_VIDEO_PATH = "video_data/match.mp4"
TARGET_VIDEO_PATH = "video_data/match_tracked_ellipses.mp4"
os.makedirs(os.path.dirname(TARGET_VIDEO_PATH), exist_ok=True)

# ---- Annotators no padrão do exemplo ----
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),  # paleta rotativa
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex("#000000"),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#FFD700"),
    base=20, height=17
)

# ---- Rastreador (ByteTrack) ----
tracker = sv.ByteTrack()
tracker.reset()

# ---- Vídeo ----
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

BALL_ID = 0  # fallback se 'class_name' não vier

with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        # confiança/overlap um pouco mais permissivos ajudam em oclusões (goleiro, etc.)
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.2, overlap=0.6)[0]
        detections = sv.Detections.from_inference(result)

        # ----- separa bola vs não-bola de forma robusta -----
        class_names = list(detections.data.get("class_name", []))
        if len(class_names) == len(detections):  # caminho preferido (por nome)
            cn = np.array(class_names, dtype=object)
            mask_ball = (cn == "ball")
        else:  # fallback por id
            mask_ball = (detections.class_id == BALL_ID)

        # bola
        ball_detections = detections[mask_ball] if len(detections) and mask_ball.any() else sv.Detections.empty()
        if len(ball_detections):
            # dá uma leve “engordada” na caixa para destacar o triângulo
            padded_xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            ball_detections = sv.Detections(
                xyxy=padded_xyxy,
                confidence=ball_detections.confidence,
                class_id=ball_detections.class_id,
                tracker_id=ball_detections.tracker_id if hasattr(ball_detections, "tracker_id") else None,
                data=ball_detections.data if hasattr(ball_detections, "data") else None,
            )

        # não-bola
        non_ball_detections = detections[~mask_ball] if len(detections) and (~mask_ball).any() else sv.Detections.empty()
        if len(non_ball_detections):
            # NMS class-agnostic (igual ao seu exemplo de 1 frame)
            non_ball_detections = non_ball_detections.with_nms(threshold=0.5, class_agnostic=True)
            # reindex opcional do exemplo (não é necessário para as anotações, mas é inofensivo)
            if hasattr(non_ball_detections, "class_id") and non_ball_detections.class_id is not None:
                non_ball_detections.class_id = non_ball_detections.class_id - 1
            # tracking
            non_ball_detections = tracker.update_with_detections(non_ball_detections)

        # ----- labels com #tracker_id -----
        labels = []
        if len(non_ball_detections) and getattr(non_ball_detections, "tracker_id", None) is not None:
            labels = [f"#{tid}" for tid in non_ball_detections.tracker_id]
        elif len(non_ball_detections):
            # fallback: sem tracker_id
            labels = ["" for _ in range(len(non_ball_detections))]

        # ----- desenha -----
        annotated = frame.copy()
        if len(non_ball_detections):
            annotated = ellipse_annotator.annotate(annotated, non_ball_detections)
            if labels:
                annotated = label_annotator.annotate(annotated, non_ball_detections, labels=labels)

        if len(ball_detections):
            annotated = triangle_annotator.annotate(annotated, ball_detections)

        video_sink.write_frame(annotated)
