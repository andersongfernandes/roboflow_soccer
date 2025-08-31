from tqdm import tqdm
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch  # (opcional, não usado no loop)
import supervision as sv
import numpy as np
import cv2
import os

# Providers do ONNX Runtime (sem aspas extras nos nomes!)
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

# --- Modelos ---
ROBOFLOW_API_KEY = "Q81j12ROpMGU5e7dBqkO"
PITCH_DETECTION_MODEL_ID = "football-field-detection-f07vi/15"
PITCH_DETECTION_MODEL = get_model(
    model_id=PITCH_DETECTION_MODEL_ID,
    api_key=ROBOFLOW_API_KEY
)

# --- Vídeo ---
SOURCE_VIDEO_PATH = "video_data/match.mp4"
TARGET_VIDEO_PATH = "video_data/match_pitch_overlay.mp4"
os.makedirs(os.path.dirname(TARGET_VIDEO_PATH), exist_ok=True)

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)

# --- Config do campo ---
CONFIG = SoccerPitchConfiguration()
PITCH_VERTICES = np.array(CONFIG.vertices, dtype=np.float32)   # [V, 2]
PITCH_EDGES = CONFIG.edges

# --- Annotators (cores do seu exemplo) ---
vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.from_hex('#FF1493'),  # rosa
    radius=8
)
edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#00BFFF'),  # azul claro
    thickness=2,
    edges=PITCH_EDGES
)

# --- Transformação de vista (homografia) ---
class ViewTransformer:
    def __init__(self):
        self.H = None  # homografia atual (3x3)

    def fit(self, source_points: np.ndarray, target_points: np.ndarray) -> bool:
        """
        Calcula homografia H tal que target ~ H * source
        source_points/target_points: shape [K, 2] float32
        return: True se conseguiu estimar
        """
        if source_points.shape[0] >= 4 and target_points.shape[0] >= 4:
            H, _ = cv2.findHomography(source_points.astype(np.float32),
                                      target_points.astype(np.float32),
                                      method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if H is not None:
                self.H = H
                return True
        return False

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Aplica H aos pontos (se existir), retornando shape [N, 2] float32
        """
        if self.H is None:
            return None
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        proj = cv2.perspectiveTransform(pts, self.H)
        return proj.reshape(-1, 2).astype(np.float32)

view = ViewTransformer()

# Parâmetros
MIN_CONF = 0.50     # confiança mínima para usar keypoint como correspondência
MIN_PTS   = 6       # pontos mínimos para atualizar homografia (>=4 é o mínimo matemático; usamos 6 para robustez)
MODEL_CONFIDENCE = 0.30
MODEL_OVERLAP    = 0.50

with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        # 1) Detecta keypoints do campo no frame
        inf = PITCH_DETECTION_MODEL.infer(frame, confidence=MODEL_CONFIDENCE, overlap=MODEL_OVERLAP)[0]
        kps = sv.KeyPoints.from_inference(inf)

        # Esperamos shape [1, V, 2] e confidence [1, V]; tratamos variações
        if hasattr(kps, "xy") and kps.xy is not None and len(kps.xy) > 0:
            xy_all = kps.xy[0]                     # [V, 2]
            conf_all = kps.confidence[0]           # [V]
        else:
            xy_all = np.zeros((0, 2), dtype=np.float32)
            conf_all = np.zeros((0,), dtype=np.float32)

        # 2) Seleciona correspondências confiáveis (pitch_vertices[i] -> xy_all[i])
        #    Assumimos que o índice do keypoint previsto corresponde ao índice do vértice no CONFIG
        if xy_all.shape[0] == PITCH_VERTICES.shape[0]:
            mask = conf_all > MIN_CONF
            src_pts = PITCH_VERTICES[mask]  # no "pitch" (fonte)
            dst_pts = xy_all[mask]          # no frame (alvo)
        else:
            # fallback: sem emparelhamento seguro (tamanho diferente), não atualiza
            src_pts = np.zeros((0, 2), dtype=np.float32)
            dst_pts = np.zeros((0, 2), dtype=np.float32)

        # 3) Atualiza homografia se houver pontos suficientes
        updated = False
        if src_pts.shape[0] >= MIN_PTS:
            updated = view.fit(source_points=src_pts, target_points=dst_pts)

        # 4) Projeta todos os vértices do campo para o frame
        overlay = frame.copy()
        proj_points = view.transform_points(PITCH_VERTICES)

        if proj_points is not None:
            # desenha as arestas do campo (todas as linhas) e os vértices usados
            frame_all_keypoints = sv.KeyPoints(xy=proj_points[np.newaxis, ...])
            overlay = edge_annotator.annotate(overlay, frame_all_keypoints)

            # desenha apenas os vértices de referência usados na homografia, se houver
            if src_pts.shape[0] > 0:
                # precisamos projetar src_pts (que estão no espaço "pitch") para o frame
                proj_used = view.transform_points(src_pts)
                used_keypoints = sv.KeyPoints(xy=proj_used[np.newaxis, ...])
                overlay = vertex_annotator.annotate(overlay, used_keypoints)

        # 5) (Opcional) Escreve info de debug no frame
        txt = f"Pitch points used: {src_pts.shape[0]} | H updated: {bool(updated)}"
        cv2.putText(overlay, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        # 6) Escreve no vídeo
        video_sink.write_frame(overlay)
