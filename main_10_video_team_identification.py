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
import torch
from transformers import AutoProcessor, SiglipVisionModel
import umap
from sklearn.cluster import KMeans
from more_itertools import chunked
import cv2

from sports.common.team import TeamClassifier
from inference import get_model

# ------------------- Configs -------------------
ROBOFLOW_API_KEY = "Q81j12ROpMGU5e7dBqkO"
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PLAYER_DETECTION_MODEL = get_model(
    model_id=PLAYER_DETECTION_MODEL_ID,
    api_key=ROBOFLOW_API_KEY
)

# vídeo de entrada/saída
SOURCE_VIDEO_PATH = "video_data/match.mp4"
TARGET_VIDEO_PATH = "video_data/match_teams_overlay.mp4"
os.makedirs(os.path.dirname(TARGET_VIDEO_PATH), exist_ok=True)

# coleta de crops para treinar o classificador
STRIDE = 30

# ids de classe do modelo
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3  # vamos ignorar na coloração de times

# embeddings (opcional para debug)
SIGLIP_VISION_PATH = 'google/siglip-base-patch16-224'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_VISION_PATH).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_VISION_PATH)
BATCH_SIZE = 32

# ------------------- Funções auxiliares -------------------
def extract_crops(source_video_path: str) -> list:
    """
    Extrai crops de jogadores (class_id=PLAYER_ID) ao longo do vídeo com stride,
    para treinar o TeamClassifier.
    """
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
) -> np.ndarray:
    """
    Atribui time a cada goleiro com base na distância aos centróides (BOTTOM_CENTER)
    dos jogadores de cada time.
    """
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

# ------------------- Treino do classificador de times -------------------
crops = extract_crops(SOURCE_VIDEO_PATH)

team_classifier = TeamClassifier(device=DEVICE)
if len(crops) > 0:
    team_classifier.fit(crops)
else:
    raise RuntimeError("Nenhum crop de jogador encontrado. Verifique o vídeo/modelo.")

# ------------------- (Opcional) Embeddings & Clusters para inspeção -------------------
# Comente este bloco se quiser acelerar.
crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
batches = chunked(crops_pil, BATCH_SIZE)
embeddings_list = []
with torch.no_grad():
    for batch in tqdm(batches, desc='embeddings extraction'):
        batch = list(batch)
        if not batch:
            continue
        inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = EMBEDDINGS_MODEL(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        embeddings_list.append(embeddings)

data = np.concatenate(embeddings_list, axis=0) if embeddings_list else np.empty((0, 768))
if data.shape[0] >= 2:
    REDUCER = umap.UMAP(n_components=3)
    CLUSTER_MODEL = KMeans(n_clusters=2)
    _ = CLUSTER_MODEL.fit(REDUCER.fit_transform(data))  # apenas para explorar offline

# ------------------- Anotadores -------------------
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),  # times 0/1, cor extra
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

# ------------------- Tracking + Vídeo -------------------
tracker = sv.ByteTrack()
tracker.reset()

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)

# cache: tracker_id -> team_id
track_team_cache = {}

with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="rendering video"):
        # 1) detecção
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        # 2) separa bola (com leve padding)
        ball_detections = detections[detections.class_id == BALL_ID]
        if len(ball_detections) > 0:
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        # 3) NMS + tracking para o resto
        non_ball = detections[detections.class_id != BALL_ID].with_nms(threshold=0.5, class_agnostic=True)
        tracked = tracker.update_with_detections(non_ball)

        # 4) separa classes
        players_detections = tracked[tracked.class_id == PLAYER_ID]
        goalkeepers_detections = tracked[tracked.class_id == GOALKEEPER_ID]
        # (opcional) árbitros – não entram na paleta de time para não confundir cores
        # referees_detections = tracked[tracked.class_id == REFEREE_ID]

        # 5) classifica time dos jogadores (com cache por tracker_id)
        if len(players_detections) > 0:
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            if len(players_crops) > 0:
                predicted = team_classifier.predict(players_crops)  # 0 ou 1
                # aplica cache por tracker_id para estabilizar
                if players_detections.tracker_id is not None:
                    new_class_ids = players_detections.class_id.copy()
                    for i, tid in enumerate(players_detections.tracker_id):
                        if tid is None:
                            new_class_ids[i] = int(predicted[i])
                        else:
                            if tid not in track_team_cache:
                                track_team_cache[tid] = int(predicted[i])
                            new_class_ids[i] = track_team_cache[tid]
                    players_detections.class_id = new_class_ids
                else:
                    players_detections.class_id = predicted.astype(int)

        # 6) time dos goleiros por centróides (dependendo dos jogadores já classificados)
        if len(goalkeepers_detections) > 0:
            gk_ids = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
            goalkeepers_detections.class_id = gk_ids

        # 7) junta (jogadores + goleiros) para anotar com mesma paleta
        all_detections = sv.Detections.merge([players_detections, goalkeepers_detections])

        # 8) labels: precisam casar com o número de detecções
        if len(all_detections) > 0 and all_detections.tracker_id is not None:
            labels = [f"#{tid}" if tid is not None else "" for tid in all_detections.tracker_id]
        else:
            labels = [""] * len(all_detections)

        # 9) render
        annotated = frame.copy()
        if len(all_detections) > 0:
            annotated = ellipse_annotator.annotate(annotated, all_detections)
        if len(ball_detections) > 0:
            annotated = triangle_annotator.annotate(annotated, ball_detections)
        if len(all_detections) > 0:
            annotated = label_annotator.annotate(annotated, all_detections, labels=labels)

        # (opcional) texto de debug
        cv2.putText(
            annotated,
            f"players:{len(players_detections)} gk:{len(goalkeepers_detections)} ball:{len(ball_detections)}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA
        )

        # 10) grava frame
        video_sink.write_frame(annotated)

print(f"✅ Vídeo gerado em: {TARGET_VIDEO_PATH}")
