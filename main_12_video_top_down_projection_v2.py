# -*- coding: utf-8 -*-
"""
Gera UM vídeo final (MP4) com a movimentação no pitch canônico, com SUAVIZAÇÃO:
- EMA de posições por track_id (players/goalies/refs)
- Histerese para troca de time (A/B) baseada em HSV
- Ball smoothing + hold de última posição
- TTL para remover tracks sumidos

Requisitos: opencv-python, numpy, supervision, scikit-learn, sua função inference.get_model
"""

# ---------------- Boot e ambiente ----------------
import os, multiprocessing, warnings
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[DmlExecutionProvider, CPUExecutionProvider]"
for var in [
    "FLORENCE2_ENABLED","QWEN_2_5_ENABLED","CORE_MODEL_SAM_ENABLED","CORE_MODEL_SAM2_ENABLED",
    "CORE_MODEL_CLIP_ENABLED","CORE_MODEL_GAZE_ENABLED","SMOLVLM2_ENABLED","DEPTH_ESTIMATION_ENABLED",
    "MOONDREAM2_ENABLED","CORE_MODEL_TROCR_ENABLED","CORE_MODEL_GROUNDINGDINO_ENABLED",
    "CORE_MODEL_YOLO_WORLD_ENABLED","CORE_MODEL_PE_ENABLED"
]:
    os.environ[var] = "False"
warnings.filterwarnings("ignore", category=UserWarning, module=r"joblib\.externals\.loky\.backend\.context")

# ---------------- Imports ----------------
import cv2
import numpy as np
from pathlib import Path
from collections import deque

import supervision as sv
from sklearn.cluster import KMeans  # leve, só para bootstrap HSV

from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

# ---------------- Modelos Roboflow ----------------
from inference import get_model

# ---------------- IO ----------------
SOURCE_VIDEO_PATH = "video_data/match.mp4"
OUTPUT_VIDEO_PATH = "output/minimapa.suave.mp4"

# ---------------- Model IDs & classes ----------------
ROBOFLOW_API_KEY = "Q81j12ROpMGU5e7dBqkO"
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
PITCH_DETECTION_MODEL_ID  = "football-field-detection-f07vi/15"

BALL_ID       = 0
GOALKEEPER_ID = 1
PLAYER_ID     = 2
REFEREE_ID    = 3

# ---------------- Pipeline base ----------------
DETECT_EVERY        = 3     # detectar a cada N frames (tracking nos intermediários)
HOMOGRAPHY_EVERY    = 20    # recalcular homografia a cada N frames
BOOTSTRAP_SECONDS   = 2.0   # segundos iniciais p/ aprender HSV dos times
CONF_DET            = 0.30
NMS_IOU             = 0.50

# ---------------- Suavização / Anti-flicker ----------------
ALPHA_POS           = 0.35  # EMA das posições no pitch (0.2-0.5)
TRACK_TTL_FRAMES    = 30    # some com track após X frames sem update
TEAM_STABILITY_FR   = 5     # nº de frames que a “nova cor” deve persistir para trocar de time
TEAM_MARGIN_MIN     = 12.0  # margem mínima |dA - dB| (em HSV) para permitir switch
BALL_HOLD_FRAMES    = 12    # manter última bola por X frames sem detecção
BALL_ALPHA          = 0.45  # EMA da bola quando detectada

# ---------------- Visual ----------------
DOT_R_PLAYER = 14
DOT_R_BALL   = 9
COLOR_TEAM_A = sv.Color.from_hex("#00BFFF")  # ciano
COLOR_TEAM_B = sv.Color.from_hex("#FF1493")  # rosa
COLOR_REF    = sv.Color.from_hex("#FFD700")  # amarelo
COLOR_BALL   = sv.Color.WHITE

PITCH_SCALE = 1.0
OUTPUT_FPS_OVERRIDE = None

# ---------------- Utils ----------------
def ensure_parent_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def central_hsv_patch(img_bgr: np.ndarray, frac: float = 0.5) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    cw, ch = int(w*frac), int(h*frac)
    x0 = (w - cw)//2
    y0 = (h - ch)//2
    patch = img_bgr[y0:y0+ch, x0:x0+cw]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3)

def median_hsv_of_crop(crop_bgr: np.ndarray) -> np.ndarray:
    hsv = central_hsv_patch(crop_bgr, frac=0.5)
    med = np.median(hsv, axis=0)
    return med

def assign_team_by_hsv_with_margin(crop_bgr: np.ndarray, cA: np.ndarray, cB: np.ndarray) -> tuple[int, float]:
    hsv_med = median_hsv_of_crop(crop_bgr)
    dA = np.linalg.norm(hsv_med - cA)
    dB = np.linalg.norm(hsv_med - cB)
    team = 0 if dA <= dB else 1
    margin = abs(dA - dB)
    return team, margin

def resolve_goalkeepers_team_id(players_detections: sv.Detections,
                                goalkeepers_detections: sv.Detections) -> np.ndarray:
    if len(goalkeepers_detections) == 0:
        return np.array([], dtype=int)
    if len(players_detections) == 0:
        return np.zeros(len(goalkeepers_detections), dtype=int)

    has_team0 = np.any(players_detections.class_id == 0)
    has_team1 = np.any(players_detections.class_id == 1)

    goalkeepers_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(goalkeepers_detections)==0 else goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy     = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    if not has_team0 and not has_team1:
        return np.zeros(len(goalkeepers_detections), dtype=int)
    if not has_team0:
        return np.ones(len(goalkeepers_detections), dtype=int)
    if not has_team1:
        return np.zeros(len(goalkeepers_detections), dtype=int)

    team_0_centroid = players_xy[players_detections.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_detections.class_id == 1].mean(axis=0)

    goalkeepers_team_ids = []
    gk_xy = goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    for goalkeeper_xy in gk_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_ids.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_ids, dtype=int)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m, _ = cv2.findHomography(source.astype(np.float32), target.astype(np.float32))
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        pts = cv2.perspectiveTransform(pts, self.m)
        return pts.reshape(-1, 2).astype(np.float32)

# ---------------- Pitch ----------------
CONFIG = SoccerPitchConfiguration()
PITCH_BASE = draw_pitch(config=CONFIG)

def compute_pitch_homography(frame_bgr: np.ndarray, model) -> ViewTransformer | None:
    result = model.infer(frame_bgr, confidence=0.30)[0]
    key_points = sv.KeyPoints.from_inference(result)
    if key_points.xy is None or len(key_points.xy) == 0:
        return None
    mask = key_points.confidence[0] > 0.5
    frame_pts = key_points.xy[0][mask]
    if frame_pts.shape[0] < 4:
        return None
    pitch_pts = np.array(CONFIG.vertices)[mask]
    try:
        return ViewTransformer(frame_pts, pitch_pts)
    except Exception:
        return None

# ---------------- Estados por track (suavização) ----------------
class TrackState:
    __slots__ = ("pos", "last_frame", "team", "flip_streak")
    def __init__(self, pos: np.ndarray, frame_idx: int, team: int):
        self.pos = pos.astype(np.float32)            # posição suavizada (x,y) no pitch
        self.last_frame = frame_idx                  # último frame com update
        self.team = team                             # time atual
        self.flip_streak = 0                         # contagem para trocar time

class TrackSmoother:
    def __init__(self, alpha_pos: float, ttl_frames: int,
                 team_stability_fr: int, team_margin_min: float):
        self.alpha = alpha_pos
        self.ttl = ttl_frames
        self.team_stability_fr = team_stability_fr
        self.team_margin_min = team_margin_min
        self.players: dict[int, TrackState] = {}
        self.goalies: dict[int, TrackState] = {}
        self.refs: dict[int, TrackState] = {}

    def _update_group(self, group: dict[int, TrackState],
                      ids: np.ndarray, xy: np.ndarray,
                      teams: list[int], margins: list[float],
                      frame_idx: int):
        for i, tid in enumerate(ids):
            if tid is None:
                continue
            tid = int(tid)
            p_new = xy[i].astype(np.float32)
            t_new = int(teams[i]) if teams is not None else 0
            margin = float(margins[i]) if margins is not None else 1e9

            if tid in group:
                st = group[tid]
                # EMA posição
                st.pos = self.alpha * p_new + (1.0 - self.alpha) * st.pos
                st.last_frame = frame_idx

                # Histerese para time
                if t_new != st.team:
                    if margin >= self.team_margin_min:
                        st.flip_streak += 1
                        if st.flip_streak >= self.team_stability_fr:
                            st.team = t_new
                            st.flip_streak = 0
                    else:
                        # margem fraca -> não conta para flip
                        st.flip_streak = 0
                else:
                    st.flip_streak = 0
            else:
                group[tid] = TrackState(pos=p_new, frame_idx=frame_idx, team=t_new)

    def update_players(self, ids, xy, teams, margins, frame_idx):
        if len(xy) > 0 and ids is not None:
            self._update_group(self.players, ids, xy, teams, margins, frame_idx)

    def update_goalies(self, ids, xy, teams, frame_idx):
        margins = [1e9] * len(xy)  # goleiro: sem margem (usa centróide)
        if len(xy) > 0 and ids is not None:
            self._update_group(self.goalies, ids, xy, teams, margins, frame_idx)

    def update_refs(self, ids, xy, frame_idx):
        teams = [0] * len(xy)      # refs não têm time; desenha como amarelo
        margins = [1e9] * len(xy)
        if len(xy) > 0 and ids is not None:
            self._update_group(self.refs, ids, xy, teams, margins, frame_idx)

    def sweep(self, frame_idx):
        # Remove tracks “velhos”
        for group in (self.players, self.goalies, self.refs):
            to_del = [tid for tid, st in group.items() if (frame_idx - st.last_frame) > self.ttl]
            for tid in to_del:
                del group[tid]

    def snapshot_arrays(self):
        # Constrói arrays para desenhar
        def group_to_arrays(group: dict[int, TrackState], include_team=True):
            if not group:
                if include_team:
                    return np.zeros((0,2), np.float32), np.zeros((0,), np.int32)
                else:
                    return np.zeros((0,2), np.float32)
            pts = np.array([st.pos for st in group.values()], dtype=np.float32)
            if include_team:
                teams = np.array([st.team for st in group.values()], dtype=np.int32)
                return pts, teams
            else:
                return pts

        players_xy, players_team = group_to_arrays(self.players, include_team=True)
        goalies_xy, goalies_team = group_to_arrays(self.goalies, include_team=True)
        refs_xy = group_to_arrays(self.refs, include_team=False)
        return players_xy, players_team, goalies_xy, goalies_team, refs_xy

# ---------------- Bola (estado simples) ----------------
class BallState:
    def __init__(self, alpha: float, hold_frames: int):
        self.alpha = alpha
        self.hold = hold_frames
        self.pos: np.ndarray | None = None
        self.last_frame = -999
    def update(self, xy: np.ndarray, frame_idx: int):
        if xy.shape[0] == 0:
            return
        p_new = xy[0].astype(np.float32)  # se houver múltiplas, pega a primeira
        if self.pos is None:
            self.pos = p_new
        else:
            self.pos = self.alpha * p_new + (1.0 - self.alpha) * self.pos
        self.last_frame = frame_idx
    def current(self, frame_idx: int):
        if self.pos is None:
            return np.zeros((0,2), dtype=np.float32)
        if (frame_idx - self.last_frame) > self.hold:
            return np.zeros((0,2), dtype=np.float32)
        return self.pos.reshape(1,2).astype(np.float32)

# ---------------- Bootstrap HSV ----------------
def bootstrap_team_hsv_centroids(capture: cv2.VideoCapture,
                                 seconds: float,
                                 detector,
                                 fps: float,
                                 conf_det: float,
                                 nms_iou: float,
                                 max_samples: int = 400) -> tuple[np.ndarray, np.ndarray] | None:
    total_frames = int(seconds * fps)
    if total_frames <= 0:
        total_frames = int(2 * fps)

    hsv_samples = []
    curr_pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    end_pos  = min(curr_pos + total_frames, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

    for _ in range(curr_pos, end_pos):
        ok, frame = capture.read()
        if not ok:
            break
        result = detector.infer(frame, confidence=conf_det)[0]
        det = sv.Detections.from_inference(result)
        det = det.with_nms(threshold=nms_iou, class_agnostic=True)
        det_players = det[det.class_id == PLAYER_ID]
        if len(det_players) == 0:
            continue
        for xyxy in det_players.xyxy:
            crop = sv.crop_image(frame, xyxy)
            med_hsv = median_hsv_of_crop(crop)
            hsv_samples.append(med_hsv)
            if len(hsv_samples) >= max_samples:
                break
        if len(hsv_samples) >= max_samples:
            break

    hsv_samples = np.array(hsv_samples, dtype=np.float32)
    if hsv_samples.shape[0] < 10:
        return None

    kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42)
    kmeans.fit(hsv_samples)
    c0, c1 = kmeans.cluster_centers_
    return (c0, c1)

# ---------------- Principal ----------------
def main():
    ensure_parent_dir(OUTPUT_VIDEO_PATH)

    # Modelos
    player_detector = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    pitch_detector  = get_model(model_id=PITCH_DETECTION_MODEL_ID,  api_key=ROBOFLOW_API_KEY)

    # Vídeo
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {SOURCE_VIDEO_PATH}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = OUTPUT_FPS_OVERRIDE if OUTPUT_FPS_OVERRIDE else in_fps

    # Canvas de saída
    pitch_img0 = PITCH_BASE.copy()
    H, W = pitch_img0.shape[:2]
    if PITCH_SCALE != 1.0:
        W = int(W * PITCH_SCALE)
        H = int(H * PITCH_SCALE)
        pitch_img0 = cv2.resize(pitch_img0, (W, H), interpolation=cv2.INTER_LINEAR)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, out_fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Não foi possível abrir o writer para: {OUTPUT_VIDEO_PATH}")

    # Bootstrap HSV
    print("[INFO] Bootstrap HSV dos times (rápido)...")
    start_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    boot = bootstrap_team_hsv_centroids(
        capture=cap,
        seconds=BOOTSTRAP_SECONDS,
        detector=player_detector,
        fps=in_fps,
        conf_det=CONF_DET,
        nms_iou=NMS_IOU,
        max_samples=400
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

    if boot is None:
        print("[AVISO] Bootstrap HSV insuficiente. Usando centróides default.")
        centroidA = np.array([100, 150, 150], dtype=np.float32)
        centroidB = np.array([320 % 180, 150, 150], dtype=np.float32)
    else:
        centroidA, centroidB = boot

    # Smoothers
    smoother = TrackSmoother(
        alpha_pos=ALPHA_POS,
        ttl_frames=TRACK_TTL_FRAMES,
        team_stability_fr=TEAM_STABILITY_FR,
        team_margin_min=TEAM_MARGIN_MIN
    )
    ball_state = BallState(alpha=BALL_ALPHA, hold_frames=BALL_HOLD_FRAMES)

    # Tracking
    tracker = sv.ByteTrack()
    tracker.reset()

    # Homografia
    vt: ViewTransformer | None = None
    last_hg_frame = -999

    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        need_detect = (f_idx % DETECT_EVERY == 0)

        if need_detect:
            infer = player_detector.infer(frame, confidence=CONF_DET)[0]
            det   = sv.Detections.from_inference(infer)
            # bola separada
            ball_det = det[det.class_id == BALL_ID]
            non_ball = det[det.class_id != BALL_ID].with_nms(threshold=NMS_IOU, class_agnostic=True)
            tracked  = tracker.update_with_detections(non_ball)
        else:
            tracked  = tracker.update_with_detections(sv.Detections.empty())
            ball_det = sv.Detections.empty()

        # Homografia (esparsa)
        if vt is None or (f_idx - last_hg_frame) >= HOMOGRAPHY_EVERY:
            vt_try = compute_pitch_homography(frame, pitch_detector)
            if vt_try is not None:
                vt = vt_try
                last_hg_frame = f_idx

        # Separa classes (tracked)
        players  = tracked[tracked.class_id == PLAYER_ID]
        goalies  = tracked[tracked.class_id == GOALKEEPER_ID]
        referees = tracked[tracked.class_id == REFEREE_ID]

        # Coordenadas no frame (anchors)
        frame_ball_xy    = ball_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(ball_det) > 0 else np.zeros((0,2))
        frame_players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(players) > 0 else np.zeros((0,2))
        frame_goalies_xy = goalies.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(goalies) > 0 else np.zeros((0,2))
        frame_refs_xy    = referees.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(referees) > 0 else np.zeros((0,2))

        # Projeção para pitch
        if vt is not None:
            pitch_ball_xy    = vt.transform_points(frame_ball_xy)
            pitch_players_xy = vt.transform_points(frame_players_xy)
            pitch_goalies_xy = vt.transform_points(frame_goalies_xy)
            pitch_refs_xy    = vt.transform_points(frame_refs_xy)
        else:
            pitch_ball_xy    = np.zeros((0,2), np.float32)
            pitch_players_xy = np.zeros((0,2), np.float32)
            pitch_goalies_xy = np.zeros((0,2), np.float32)
            pitch_refs_xy    = np.zeros((0,2), np.float32)

        # Times dos players por HSV (com margem/histerese)
        player_teams = []
        player_margins = []
        if len(players) > 0 and pitch_players_xy.shape[0] > 0:
            # crops a partir das bbox no frame
            crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
            for crop in crops:
                t, m = assign_team_by_hsv_with_margin(crop, centroidA, centroidB)
                player_teams.append(t)
                player_margins.append(m)
        # Goleiros pelo centróide do time
        goalie_teams = []
        if len(goalies) > 0:
            goalie_teams = resolve_goalkeepers_team_id(
                players_detections=players if len(players)>0 else sv.Detections.empty(),
                goalkeepers_detections=goalies
            ).tolist()

        # Atualiza smoothers (IDs do tracker)
        pid = players.tracker_id if hasattr(players, "tracker_id") else None
        gid = goalies.tracker_id if hasattr(goalies, "tracker_id") else None
        rid = referees.tracker_id if hasattr(referees, "tracker_id") else None

        if pid is not None and len(pid) != 0:
            smoother.update_players(ids=pid, xy=pitch_players_xy,
                                    teams=player_teams if player_teams else [0]*len(pitch_players_xy),
                                    margins=player_margins if player_margins else [1e9]*len(pitch_players_xy),
                                    frame_idx=f_idx)
        if gid is not None and len(gid) != 0:
            smoother.update_goalies(ids=gid, xy=pitch_goalies_xy,
                                    teams=goalie_teams if goalie_teams else [0]*len(pitch_goalies_xy),
                                    frame_idx=f_idx)
        if rid is not None and len(rid) != 0:
            smoother.update_refs(ids=rid, xy=pitch_refs_xy, frame_idx=f_idx)

        # Bola (EMA + hold)
        if pitch_ball_xy.shape[0] > 0:
            ball_state.update(pitch_ball_xy, f_idx)
        ball_xy_smooth = ball_state.current(f_idx)

        # Expira tracks velhos
        smoother.sweep(f_idx)

        # Snapshot para desenho
        players_xy, players_team, goalies_xy, goalies_team, refs_xy = smoother.snapshot_arrays()

        # Canvas
        canvas = pitch_img0.copy()

        # Bola
        if len(ball_xy_smooth) > 0:
            canvas = draw_points_on_pitch(
                config=CONFIG, xy=ball_xy_smooth,
                face_color=COLOR_BALL, edge_color=sv.Color.BLACK,
                radius=DOT_R_BALL, pitch=canvas
            )
        # Players A/B
        if len(players_xy) > 0:
            mA = (players_team == 0)
            mB = (players_team == 1)
            if np.any(mA):
                canvas = draw_points_on_pitch(
                    config=CONFIG, xy=players_xy[mA],
                    face_color=COLOR_TEAM_A, edge_color=sv.Color.BLACK,
                    radius=DOT_R_PLAYER, pitch=canvas
                )
            if np.any(mB):
                canvas = draw_points_on_pitch(
                    config=CONFIG, xy=players_xy[mB],
                    face_color=COLOR_TEAM_B, edge_color=sv.Color.BLACK,
                    radius=DOT_R_PLAYER, pitch=canvas
                )
        # Goleiros (poderia desenhar com contorno especial; aqui seguem cor do time)
        if len(goalies_xy) > 0:
            mA = (goalies_team == 0)
            mB = (goalies_team == 1)
            if np.any(mA):
                canvas = draw_points_on_pitch(
                    config=CONFIG, xy=goalies_xy[mA],
                    face_color=COLOR_TEAM_A, edge_color=sv.Color.BLACK,
                    radius=DOT_R_PLAYER, pitch=canvas
                )
            if np.any(mB):
                canvas = draw_points_on_pitch(
                    config=CONFIG, xy=goalies_xy[mB],
                    face_color=COLOR_TEAM_B, edge_color=sv.Color.BLACK,
                    radius=DOT_R_PLAYER, pitch=canvas
                )
        # Árbitros
        if len(refs_xy) > 0:
            canvas = draw_points_on_pitch(
                config=CONFIG, xy=refs_xy,
                face_color=COLOR_REF, edge_color=sv.Color.BLACK,
                radius=DOT_R_PLAYER, pitch=canvas
            )

        writer.write(canvas)
        f_idx += 1

    writer.release()
    cap.release()
    print(f"[OK] Vídeo suave gerado em: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()

