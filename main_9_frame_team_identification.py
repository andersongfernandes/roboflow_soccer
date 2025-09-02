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

# Agora sim, os demais imports
from tqdm import tqdm
import supervision as sv
import numpy as np
from sports.common.team import TeamClassifier

import torch
from transformers import AutoProcessor, SiglipVisionModel
import umap
from sklearn.cluster import KMeans
from more_itertools import chunked

# ------------------- Configs -------------------
from inference import get_model
ROBOFLOW_API_KEY = "Q81j12ROpMGU5e7dBqkO"
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PLAYER_DETECTION_MODEL = get_model(
    model_id=PLAYER_DETECTION_MODEL_ID,
    api_key=ROBOFLOW_API_KEY
)

STRIDE = 30
SOURCE_VIDEO_PATH = "video_data/match.mp4"

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3  # (não usado aqui)

SIGLIP_VISION_PATH = 'google/siglip-base-patch16-224'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_VISION_PATH).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_VISION_PATH)

BATCH_SIZE = 32

# ------------------- Funções -------------------
def extract_crops(source_video_path: str):
    frame_generator = sv.get_video_frames_generator(source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        results = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(results)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    return crops

def resolve_goalkeepers_team_id(
    players_detections: sv.Detections,
    goalkeepers_detections: sv.Detections
):
    if len(goalkeepers_detections) == 0:
        return np.array([], dtype=int)
    if len(players_detections) == 0:
        # Se não há jogadores classificados ainda, atribui tudo ao time 0 por padrão
        return np.zeros(len(goalkeepers_detections), dtype=int)

    # Verifica presença de cada time
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

# ------------------- Embeddings & Clusters -------------------
crops = extract_crops(SOURCE_VIDEO_PATH)

team_classifier = TeamClassifier(device=DEVICE)
if len(crops) > 0:
    team_classifier.fit(crops)
else:
    raise RuntimeError("Nenhum crop de jogador encontrado. Verifique o vídeo/modelo.")

# Extrai embeddings com SigLIP para clusterização
crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
batches = chunked(crops_pil, BATCH_SIZE)
embeddings_list = []

with torch.no_grad():
    for batch in tqdm(batches, desc='embeddings extraction'):
        batch = list(batch)
        if len(batch) == 0:
            continue
        inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = EMBEDDINGS_MODEL(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        embeddings_list.append(embeddings)

data = np.concatenate(embeddings_list, axis=0) if len(embeddings_list) else np.empty((0, 768))
if data.shape[0] >= 2:
    REDUCER = umap.UMAP(n_components=3)
    CLUSTER_MODEL = KMeans(n_clusters=2)
    projections = REDUCER.fit_transform(data)
    clusters = CLUSTER_MODEL.fit_predict(projections)
else:
    clusters = np.zeros((len(crops_pil),), dtype=int)

# ------------------- Tracking & Anotação do primeiro frame -------------------
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
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

tracker = sv.ByteTrack()
tracker.reset()

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
detections = sv.Detections.from_inference(result)

# Bola
ball_detections = detections[detections.class_id == BALL_ID]
if len(ball_detections) > 0:
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

# Demais (NMS + tracking)
non_ball = detections[detections.class_id != BALL_ID].with_nms(threshold=0.5, class_agnostic=True)
tracked = tracker.update_with_detections(non_ball)

# Separa por classes
players_detections = tracked[tracked.class_id == PLAYER_ID]
goalkeepers_detections = tracked[tracked.class_id == GOALKEEPER_ID]
referees_detections = tracked[tracked.class_id == REFEREE_ID]

# Predição de time para jogadores
if len(players_detections) > 0:
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    if len(players_crops) > 0:
        players_detections.class_id = team_classifier.predict(players_crops)

# Predição de time para goleiros (usando centróides)
if len(goalkeepers_detections) > 0:
    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
        players_detections, goalkeepers_detections
    )

# Predição de arbitro
if len(referees_detections) > 0:
    referees_detections.class_id = -1

# Junta jogadores + goleiros para anotar com mesma paleta
all_detections = sv.Detections.merge([players_detections, goalkeepers_detections])

# -------- FIX: labels devem ter MESMO tamanho de all_detections --------
if len(all_detections) > 0 and all_detections.tracker_id is not None:
    labels = [f"#{tid}" if tid is not None else "" for tid in all_detections.tracker_id]
else:
    labels = [""] * len(all_detections)

# ------------------- Render -------------------
annotated_frame = frame.copy()

if len(all_detections) > 0:
    annotated_frame = ellipse_annotator.annotate(annotated_frame, all_detections)

if len(ball_detections) > 0:
    annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)

if len(all_detections) > 0:
    annotated_frame = label_annotator.annotate(annotated_frame, all_detections, labels=labels)

sv.plot_image(annotated_frame)
