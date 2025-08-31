from tqdm import tqdm
import supervision as sv
import numpy as np
from sports.common.team import TeamClassifier
import os, multiprocessing

# Força backends do onnxruntime (se estiver usando DirectML no Windows)
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[DmlExecutionProvider, CPUExecutionProvider]"

# Evita warning do joblib/loky sobre núcleos físicos
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(multiprocessing.cpu_count()))


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
STRIDE = 30
SOURCE_VIDEO_PATH = "video_data/match.mp4"
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFERRE_ID = 3

def extract_crops(source_video_path: str):
    frame_generator = sv.get_video_frames_generator(source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        results = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(results)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        crops += [
            sv.crop_image(frame, xyxy)
            for xyxy
            in detections.xyxy
        ]

    return crops

# sv.plot_images_grid(crops[:100], grid_size=(10,10))

import torch
from transformers import AutoProcessor, SiglipVisionModel

SIGLIP_VISION_PATH = 'google/siglip-base-patch16-224'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_VISION_PATH).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_VISION_PATH)

crops = extract_crops(SOURCE_VIDEO_PATH)
team_classifier = TeamClassifier(device=DEVICE)

from more_itertools import chunked
BATCH_SIZE = 32

crops = [sv.cv2_to_pillow(crop) for crop in crops]
batches = chunked(crops, BATCH_SIZE)
data = []
embeddings_list = []

with torch.no_grad():
    for batch in tqdm(batches, desc='embeddings extraction'):
        # garanta que 'batch' é lista (more_itertools.chunked gera iterável)
        batch = list(batch)

        # 1) parâmetro correto é return_tensors
        inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors='pt')

        # 2) mover cada tensor do dict para o device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # 3) forward; para modelos vision, a chave é 'pixel_values'
        outputs = EMBEDDINGS_MODEL(**inputs)  # equivalente a EMBEDDINGS_MODEL(pixel_values=inputs['pixel_values'])

        # 4) média sobre tokens -> vetor por imagem
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        embeddings_list.append(embeddings)

data = np.concatenate(embeddings_list, axis=0)

import umap
from sklearn.cluster import KMeans

REDUCER = umap.UMAP(n_components=3)
CLUSTER_MODEL = KMeans(n_clusters=2)

projections = REDUCER.fit_transform(data)
clusters = CLUSTER_MODEL.fit_predict(projections)

team_0 = [
    crop
    for crop, cluster
    in zip(crops, clusters)
    if cluster == 0
]

sv.plot_images_grid(team_0[:100],grid_size=(10,10))