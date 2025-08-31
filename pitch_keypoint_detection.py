import supervision as sv
import numpy as np
import os
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[DmlExecutionProvider, CPUExecutionProvider]"


# Desliga os modelos opcionais que exigem transformers/torch
for var in [
    "FLORENCE2_ENABLED","QWEN_2_5_ENABLED","CORE_MODEL_SAM_ENABLED","CORE_MODEL_SAM2_ENABLED",
    "CORE_MODEL_CLIP_ENABLED","CORE_MODEL_GAZE_ENABLED","SMOLVLM2_ENABLED","DEPTH_ESTIMATION_ENABLED",
    "MOONDREAM2_ENABLED","CORE_MODEL_TROCR_ENABLED","CORE_MODEL_GROUNDINGDINO_ENABLED",
    "CORE_MODEL_YOLO_WORLD_ENABLED","CORE_MODEL_PE_ENABLED"
]:
    os.environ[var] = "False"

SOURCE_VIDEO_PATH = "video_data/match.mp4"

from inference import get_model
ROBOFLOW_API_KEY = "Q81j12ROpMGU5e7dBqkO"
PITCH_DETECTION_MODEL_ID = "football-field-detection-f07vi/15"
PITCH_DETECTION_MODEL = get_model(PITCH_DETECTION_MODEL_ID, ROBOFLOW_API_KEY)

vertex_annotator = sv.VertexAnnotator(
    color = sv.Color.from_hex('#FF1493'),
    radius = 8
)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = PITCH_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
key_points = sv.KeyPoints.from_inference(result)

filter = key_points.confidence[0] > 0.5
frame_reference_points = key_points.xy[0][filter]
frame_reference_key_points = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])

annotated_frame = frame.copy()
annotated_frame = vertex_annotator.annotate(annotated_frame, frame_reference_key_points)

sv.plot_image(annotated_frame)
