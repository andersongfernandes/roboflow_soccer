from tqdm import tqdm
import supervision as sv
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


from inference import get_model
ROBOFLOW_API_KEY = "Q81j12ROpMGU5e7dBqkO"
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PLAYER_DETECTION_MODEL = get_model(
    model_id=PLAYER_DETECTION_MODEL_ID,
    api_key=ROBOFLOW_API_KEY
)
SOURCE_VIDEO_PATH = "video_data/match.mp4"
BALL_ID = 0

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
# label_annotator = sv.LabelAnnotator(
#     color=sv.ColorPalette.from_hex(['#FF8C00','#00BFFF', '#FF1493', '#FFD700']),
#     text_color=sv.Color.from_hex("#000000")    
# )
triangle_annotator =  sv.TriangleAnnotator(
    color=sv.Color.from_hex("#FFD700"),
    base=20, height=17
)


frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
detections = sv.Detections.from_inference(result)

ball_detections = detections[detections.class_id == BALL_ID]
ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

all_detections = detections[detections.class_id != BALL_ID]
all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
all_detections.class_id = all_detections.class_id - 1

# labels = [
#     f"{class_name} {confidence:.2f}"
#     for class_name, confidence
#     in zip(detections["class_name"],detections.confidence)
# ]

annotated_frame = frame.copy()
annotated_frame = ellipse_annotator.annotate(annotated_frame, all_detections)
annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)
sv.plot_image(annotated_frame)
