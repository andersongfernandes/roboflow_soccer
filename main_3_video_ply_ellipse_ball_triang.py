from tqdm import tqdm
import supervision as sv
import numpy as np
import os

# Providers do ONNX Runtime
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

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# Anotadores por classe
ellipse_ann_player     = sv.EllipseAnnotator(color=sv.Color.from_hex("#00BFFF"), thickness=2)  # player
ellipse_ann_goalkeeper = sv.EllipseAnnotator(color=sv.Color.from_hex("#FF4500"), thickness=2)  # GK
ellipse_ann_referee    = sv.EllipseAnnotator(color=sv.Color.from_hex("#FFD700"), thickness=2)  # árbitro

triangle_ann_ball = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=20, height=17)

video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.2, overlap=0.6)[0]
        detections = sv.Detections.from_inference(result)

        # nomes das classes como lista
        class_names = list(detections.data.get("class_name", []))
        if len(class_names) != len(detections):
            class_names = ["unknown"] * len(detections)
        cn = np.array(class_names, dtype=object)

        mask_ball       = (cn == "ball")
        mask_player     = (cn == "player")
        mask_goalkeeper = (cn == "goalkeeper")
        mask_referee    = (cn == "referee")

        det_ball       = detections[mask_ball]       if len(detections) and mask_ball.any() else sv.Detections.empty()
        det_player     = detections[mask_player]     if len(detections) and mask_player.any() else sv.Detections.empty()
        det_goalkeeper = detections[mask_goalkeeper] if len(detections) and mask_goalkeeper.any() else sv.Detections.empty()
        det_referee    = detections[mask_referee]    if len(detections) and mask_referee.any() else sv.Detections.empty()

        annotated = frame.copy()

        if len(det_player):
            annotated = ellipse_ann_player.annotate(annotated, det_player)
        if len(det_goalkeeper):
            annotated = ellipse_ann_goalkeeper.annotate(annotated, det_goalkeeper)
        if len(det_referee):
            annotated = ellipse_ann_referee.annotate(annotated, det_referee)
        if len(det_ball):
            # recria Detections com xyxy acolchoado (não usa .copy())
            padded_xyxy = sv.pad_boxes(xyxy=det_ball.xyxy, px=10)
            padded_ball = sv.Detections(
                xyxy=padded_xyxy,
                confidence=det_ball.confidence,
                class_id=det_ball.class_id,
                tracker_id=det_ball.tracker_id if hasattr(det_ball, "tracker_id") else None,
                data=det_ball.data if hasattr(det_ball, "data") else None,
            )
            annotated = triangle_ann_ball.annotate(annotated, padded_ball)

        video_sink.write_frame(annotated)
