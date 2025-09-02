#  horario de inicio do execute: 21h45

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
from sports.common.team import TeamClassifier
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

import torch
from transformers import AutoProcessor, SiglipVisionModel
import umap
from sklearn.cluster import KMeans
from more_itertools import chunked

import cv2

# ------------------- Configs -------------------
from inference import get_model
ROBOFLOW_API_KEY = "Q81j12ROpMGU5e7dBqkO"

PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PLAYER_DETECTION_MODEL = get_model(
    model_id=PLAYER_DETECTION_MODEL_ID,
    api_key=ROBOFLOW_API_KEY
)

PITCH_DETECTION_MODEL_ID = "football-field-detection-f07vi/15"
PITCH_DETECTION_MODEL = get_model(
    model_id=PITCH_DETECTION_MODEL_ID,
    api_key=ROBOFLOW_API_KEY
)

# Caminhos de I/O
STRIDE = 30                        # processar 1 a cada N frames
SOURCE_VIDEO_PATH = "video_data/match.mp4"
OUTPUT_VIDEO_PATH = "outputs/match_pitch_positions.mp4"
os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

# Classes
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3  # (não usado diretamente na classificação de time)

# SigLIP (opcional, para embeddings/clusterização dos crops coletados)
SIGLIP_VISION_PATH = 'google/siglip-base-patch16-224'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_VISION_PATH).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_VISION_PATH)

BATCH_SIZE = 32
ENABLE_EMBEDDING_CLUSTERING = False  # deixe True se quiser reproduzir sua etapa de UMAP+KMeans

# Homografia: re-detectar o campo a cada N frames (para acompanhar pan/zoom)
PITCH_REDETECT_EVERY = 10
MIN_KP_CONF = 0.5  # confiança mínima dos keypoints do campo

# ------------------- Utilitários -------------------
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        # Necessita ao menos 4 pares de pontos
        if source.shape[0] < 4 or target.shape[0] < 4:
            raise ValueError("Pontos insuficientes para estimar homografia.")
        self.m, _ = cv2.findHomography(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points.astype(np.float32).reshape(-1, 2)
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)

# ------------------- Funções de pipeline -------------------
def extract_crops(source_video_path: str, stride: int) -> list:
    """Varre o vídeo coletando crops de jogadores para treinar o TeamClassifier."""
    frame_generator = sv.get_video_frames_generator(source_video_path, stride=stride)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        results = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(results)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        if len(detections) > 0:
            crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    return crops

def resolve_goalkeepers_team_id(
    players_detections: sv.Detections,
    goalkeepers_detections: sv.Detections
) -> np.ndarray:
    """Atribui time ao(s) goleiro(s) aproximando pelo centróide dos times."""
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

def compute_view_transformer(frame: np.ndarray, min_conf: float = MIN_KP_CONF):
    """Detecta o campo no frame e estima a homografia para o pitch canônico."""
    result = PITCH_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)
    if key_points.xy.shape[0] == 0:
        return None, None

    # Filtra os pontos por confiança
    filter_mask = key_points.confidence[0] > min_conf
    frame_reference_points = key_points.xy[0][filter_mask]

    if frame_reference_points.shape[0] < 4:
        return None, None

    from sports.configs.soccer import SoccerPitchConfiguration
    CONFIG = SoccerPitchConfiguration()
    pitch_reference_points = np.array(CONFIG.vertices)[filter_mask]

    try:
        vt = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )
        return vt, CONFIG
    except Exception:
        return None, None

def draw_pitch_frame(config,
                     pitch_ball_xy: np.ndarray,
                     pitch_players_xy: np.ndarray,
                     players_team_ids: np.ndarray,
                     pitch_referees_xy: np.ndarray) -> np.ndarray:
    """Gera a imagem do campo top-down com os pontos plotados."""
    pitch = draw_pitch(config=config)

    # Bola
    if pitch_ball_xy is not None and pitch_ball_xy.size > 0:
        pitch = draw_points_on_pitch(
            config=config,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=pitch
        )
    # Jogadores time 0
    if pitch_players_xy is not None and pitch_players_xy.size > 0:
        mask0 = (players_team_ids == 0)
        if np.any(mask0):
            pitch = draw_points_on_pitch(
                config=config,
                xy=pitch_players_xy[mask0],
                face_color=sv.Color.from_hex("#00BFFF"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch
            )
        # Jogadores time 1
        mask1 = (players_team_ids == 1)
        if np.any(mask1):
            pitch = draw_points_on_pitch(
                config=config,
                xy=pitch_players_xy[mask1],
                face_color=sv.Color.from_hex("#FF1493"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch
            )
    # Árbitros
    if pitch_referees_xy is not None and pitch_referees_xy.size > 0:
        pitch = draw_points_on_pitch(
            config=config,
            xy=pitch_referees_xy,
            face_color=sv.Color.from_hex("#FFD700"),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=pitch
        )

    return pitch

def maybe_cluster_embeddings(crops: list):
    """Etapa opcional de embeddings + UMAP + KMeans (não usada no vídeo)."""
    if not crops:
        return None
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
        reducer = umap.UMAP(n_components=3)
        cluster_model = KMeans(n_clusters=2, n_init=10)
        _ = cluster_model.fit(reducer.fit_transform(data))
    return None

# ------------------- Main (gera vídeo) -------------------
def main():
    # FPS do vídeo de origem
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {SOURCE_VIDEO_PATH}")
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or np.isnan(fps_in) or fps_in <= 0:
        fps_in = 30.0
    cap.release()

    # Pré-coleta de crops para treinar o classificador de time
    crops = extract_crops(SOURCE_VIDEO_PATH, stride=STRIDE)
    if len(crops) == 0:
        raise RuntimeError("Nenhum crop de jogador encontrado. Verifique o vídeo/modelo.")

    team_classifier = TeamClassifier(device=DEVICE)
    team_classifier.fit(crops)

    # (Opcional) embeddings/clusterização — mantido para compatibilidade com seu código
    if ENABLE_EMBEDDING_CLUSTERING:
        maybe_cluster_embeddings(crops)

    # Tracker
    tracker = sv.ByteTrack()
    tracker.reset()

    # Preparação da escrita de vídeo (pitch tem tamanho fixo desde o primeiro frame válido)
    # Obtemos um primeiro vt/config válido para dimensionar a saída
    print("Estimando homografia inicial do campo...")
    init_frame = next(sv.get_video_frames_generator(SOURCE_VIDEO_PATH), None)
    if init_frame is None:
        raise RuntimeError("Não foi possível ler frames do vídeo.")
    vt, CONFIG = compute_view_transformer(init_frame, min_conf=MIN_KP_CONF)
    if vt is None or CONFIG is None:
        # Vamos tentar alguns frames à frente antes de desistir
        for f in sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=max(1, STRIDE//2)):
            vt, CONFIG = compute_view_transformer(f, min_conf=MIN_KP_CONF)
            if vt is not None:
                break

    # Mesmo que vt ainda seja None, vamos inicializar um pitch "em branco" para abrir o writer
    tmp_pitch = draw_pitch(config=CONFIG) if CONFIG is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
    ph, pw = tmp_pitch.shape[:2]

    # FPS de saída proporcional ao STRIDE
    fps_out = max(1, int(round(fps_in / STRIDE)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_out, (pw, ph))

    # Loop principal de processamento -> 1 frame de saída por STRIDE frames do vídeo original
    frame_iter = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    last_vt = vt
    last_config = CONFIG

    print("Gerando vídeo do posicionamento no campo...")
    for idx, frame in enumerate(tqdm(frame_iter, desc="processing video frames")):
        # Processa apenas 1 a cada STRIDE frames
        if idx % STRIDE != 0:
            continue

        # Reestima homografia periodicamente ou se ainda não temos uma válida
        if (last_vt is None) or (idx % PITCH_REDETECT_EVERY == 0):
            vt_try, cfg_try = compute_view_transformer(frame, min_conf=MIN_KP_CONF)
            if vt_try is not None and cfg_try is not None:
                last_vt, last_config = vt_try, cfg_try

        # Detecção
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
        players_team_ids = np.array([], dtype=int)
        if len(players_detections) > 0:
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            if len(players_crops) > 0:
                players_team_ids = team_classifier.predict(players_crops)
                # Goleiros pelo centróide
                if len(goalkeepers_detections) > 0:
                    gk_team = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
                else:
                    gk_team = np.array([], dtype=int)
            else:
                gk_team = np.array([], dtype=int)
        else:
            gk_team = np.array([], dtype=int)

        # Transformação para o pitch
        if last_vt is not None and last_config is not None:
            # Anchors dos detecionados
            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(ball_detections) else np.empty((0,2), dtype=np.float32)
            frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(players_detections) else np.empty((0,2), dtype=np.float32)
            frame_referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(referees_detections) else np.empty((0,2), dtype=np.float32)

            # Projeta para o campo canônico
            pitch_ball_xy = last_vt.transform_points(frame_ball_xy) if frame_ball_xy.size else np.empty((0,2), dtype=np.float32)
            pitch_players_xy = last_vt.transform_points(frame_players_xy) if frame_players_xy.size else np.empty((0,2), dtype=np.float32)
            pitch_referees_xy = last_vt.transform_points(frame_referees_xy) if frame_referees_xy.size else np.empty((0,2), dtype=np.float32)

            # Desenha o frame do pitch
            pitch_img = draw_pitch_frame(
                config=last_config,
                pitch_ball_xy=pitch_ball_xy,
                pitch_players_xy=pitch_players_xy,
                players_team_ids=players_team_ids if players_team_ids.size else np.array([], dtype=int),
                pitch_referees_xy=pitch_referees_xy
            )
        else:
            # Sem homografia válida: apenas campo "limpo"
            pitch_img = draw_pitch(config=last_config) if last_config is not None else np.zeros((ph, pw, 3), dtype=np.uint8)

        # Overlay com info básica (frame / tempo aproximado)
        t_seconds = idx / fps_in
        overlay = pitch_img.copy()
        cv2.putText(
            overlay,
            f"frame={idx}  t={t_seconds:05.2f}s  stride={STRIDE}  fps_out~{fps_out}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv2.LINE_AA
        )
        cv2.putText(
            overlay,
            f"frame={idx}  t={t_seconds:05.2f}s  stride={STRIDE}  fps_out~{fps_out}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Escreve no vídeo (garante BGR)
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        if (bgr.shape[1], bgr.shape[0]) != (pw, ph):
            bgr = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_LINEAR)
        writer.write(bgr)

    writer.release()
    print(f"Vídeo gerado em: {OUTPUT_VIDEO_PATH}")

# ------------------- Entrypoint -------------------
if __name__ == "__main__":
    main()
