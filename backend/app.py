import os
import io
import re
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. CONSTANTS & MODEL DEFINITION
# ==============================================================================

CLASSES = [
    "Brown spot",
    "Healthy Paddy",
    "Hispa",
    "Leaf Blast"
]

DISEASE_METADATA = {
    "Healthy Paddy": {
        "status": "Healthy",
        "severity": "None",
        "pathogen": "None (Plant is physiologically healthy)",
        "symptoms": "Leaves display vibrant, uniform green coloration without necrotic lesions, discolored margins, or feeding trails.",
        "management": "Maintain standard water leveling (2-5 cm during vegetative stage), balanced NPK fertilization, and periodic field scouting.",
        "color_theme": "#16a34a"
    },
    "Brown spot": {
        "status": "Diseased",
        "severity": "Moderate to High",
        "pathogen": "Bipolaris oryzae (Cochliobolus miyabeanus)",
        "symptoms": "Small, circular to oval dark brown spots with yellow halos across leaf surfaces. Severe infection leads to seedling blight and kernel discoloration.",
        "management": "Apply balanced soil fertilization especially potassium (K) and silicon. Seed treatment with Carbendazim (2g/kg) or foliar spray of Mancozeb / Propiconazole.",
        "color_theme": "#d97706"
    },
    "Hispa": {
        "status": "Diseased",
        "severity": "Moderate to Severe",
        "pathogen": "Dicladispa armigera (Coleoptera: Chrysomelidae)",
        "symptoms": "Adult beetles scrape upper leaf epidermis creating characteristic white parallel streaks; grubs mine inside leaves causing blotch blister mines and withered tips.",
        "management": "Clip leaf tips before transplanting to destroy eggs. Sweep net collection in early morning. Foliar spray of Chlorpyrifos 20% EC (2.5 ml/L) or Quinalphos 25% EC.",
        "color_theme": "#ca8a04"
    },
    "Leaf Blast": {
        "status": "Diseased",
        "severity": "Critical / Severe",
        "pathogen": "Magnaporthe oryzae (Pyricularia oryzae)",
        "symptoms": "Spindle-shaped or diamond-shaped lesions with gray-whitish centers and dark reddish-brown margins. Rapid coalescence leads to complete leaf blighting.",
        "management": "Avoid excessive nitrogen application. Spray Tricyclazole 75% WP (0.6 g/L) or Isoprothiolane 40% EC (1.5 ml/L) at initial lesion detection. Ensure good water drainage.",
        "color_theme": "#dc2626"
    }
}

IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


class DualViewEfficientNetB0(nn.Module):
    """
    Dual-stream Siamese Neural Network using Pretrained EfficientNet-B0:
    - Extracts 1280 features from Top View (T1)
    - Extracts 1280 features from Bottom View (D1 / B1)
    - Fuses features via concatenation (2560 dimensions)
    - Non-linear classifier head maps to 4 paddy disease classes
    """
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super().__init__()
        base_model = efficientnet_b0(weights=None)

        # Shared feature extractor
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        backbone_features = base_model.classifier[1].in_features  # 1280

        # Feature Fusion Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(backbone_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(512, num_classes)
        )

    def extract_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, img_top, img_bottom=None):
        if img_bottom is None:
            img_bottom = img_top  # Siamese single-view fallback from author's training code

        feat_top = self.extract_features(img_top)        # [B, 1280]
        feat_bottom = self.extract_features(img_bottom)  # [B, 1280]

        # Feature Fusion
        fused = torch.cat([feat_top, feat_bottom], dim=1)  # [B, 2560]
        logits = self.classifier(fused)                    # [B, num_classes]
        return logits


# ==============================================================================
# 2. MODEL LOADER & GLOBAL STATE
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MODEL_PATH = BASE_DIR / "models" / "best_dual_efficientnet_b0.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[DualViewEfficientNetB0] = None
model_load_status = {"loaded": False, "error": None, "checkpoint": str(MODEL_PATH)}

def init_model():
    global model, model_load_status
    try:
        logger.info(f"Loading checkpoint from: {MODEL_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {MODEL_PATH}")

        net = DualViewEfficientNetB0(num_classes=len(CLASSES))
        checkpoint = torch.load(str(MODEL_PATH), map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Remove any unexpected prefixes if present
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        load_result = net.load_state_dict(clean_sd, strict=True)
        logger.info(f"Model state_dict loaded successfully: {load_result}")

        net.to(device)
        net.eval()
        model = net
        model_load_status = {
            "loaded": True,
            "error": None,
            "checkpoint": MODEL_PATH.name,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "classes": CLASSES,
            "total_parameters": sum(p.numel() for p in net.parameters()),
            "input_resolution": f"{IMG_SIZE}x{IMG_SIZE} RGB"
        }
        logger.info(f"Model ready on device: {device} ({model_load_status.get('device_name')})")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        model_load_status = {"loaded": False, "error": str(e), "checkpoint": str(MODEL_PATH)}


# Initialize model on module load
init_model()

# ==============================================================================
# 3. HELPER UTILITIES
# ==============================================================================

def verify_paddy_leaf_image(file_bytes: bytes) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates whether uploaded image bytes contain an authentic paddy leaf specimen.
    Performs multi-layer verification:
    1. Integrity check: file existence, byte decoding, format, minimum resolution (>=32x32).
    2. Blank / Monochrome / Random noise detection.
    3. Foliar botanical color segmentation (chlorophyll vegetative tissue, chlorotic tissue, foliar lesions).
    4. Botanical vegetative dominance and center leaf subject verification.
    
    Returns:
        (is_valid: bool, user_message: str, details: dict)
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, "Invalid image: File is empty (0 bytes). Please upload a paddy leaf image for disease detection.", {"error_type": "empty_file"}

    try:
        im = Image.open(io.BytesIO(file_bytes))
        im.verify()
        im = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        logger.warning(f"Image decode failed: {e}")
        return False, "Corrupted or invalid image: The uploaded file cannot be processed as an image. Please upload a valid image file (JPG, PNG, WEBP).", {"error_type": "corrupted_image", "raw_error": str(e)}

    w, h = im.size
    if w < 32 or h < 32:
        return False, "Invalid image: Image resolution is too small (minimum 32x32 pixels required). Please upload a clearer paddy leaf image.", {"error_type": "too_small", "width": w, "height": h}

    if max(w, h) / max(min(w, h), 1) > 25.0:
        return False, "Invalid image: Degenerate aspect ratio. Please upload a standard photograph of a paddy leaf.", {"error_type": "extreme_aspect_ratio"}

    # Resize to standard evaluation resolution (224x224) for consistent, high-speed metric calculation
    im_scaled = im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(im_scaled, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Layer 1: Blank / Solid Color / Monochrome Screen Detection
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    std_gray = float(gray.std())
    std_r = float(r.std())
    std_g = float(g.std())
    std_b = float(b.std())

    if std_gray < 5.0 or (std_r < 3.0 and std_g < 3.0 and std_b < 3.0):
        return False, "Invalid image: Please upload a paddy leaf image for disease detection.", {"error_type": "monochrome", "std_gray": std_gray}

    if float(gray.mean()) > 252.0 and std_gray < 8.0:
        return False, "Invalid image: Please upload a paddy leaf image for disease detection.", {"error_type": "blank_white"}

    if float(gray.mean()) < 5.0 and std_gray < 8.0:
        return False, "Invalid image: Please upload a paddy leaf image for disease detection.", {"error_type": "blank_black"}

    # Layer 2: Random Noise / Decorrelated Pixel Detection
    dy = np.abs(gray[1:, :] - gray[:-1, :])
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    mean_grad = float((dy.mean() + dx.mean()) / 2.0)
    g_arr = arr[:, :, 1]
    corr_matrix = np.corrcoef(g_arr[:-1, :].flatten(), g_arr[1:, :].flatten())
    spatial_corr = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]) else 0.0

    if mean_grad > 35.0 or spatial_corr < 0.60:
        return False, "Invalid image: Please upload a paddy leaf image for disease detection.", {"error_type": "noise_or_uncorrelated", "spatial_corr": spatial_corr, "mean_grad": mean_grad}

    # Layer 3: Botanical Foliar Feature Analysis (HSV & Vegetative Chromaticity)
    im_hsv = np.array(im_scaled.convert("HSV"), dtype=np.float32)
    hue, sat, val = im_hsv[:, :, 0], im_hsv[:, :, 1], im_hsv[:, :, 2]

    # 1. Green vegetative foliar tissue (chlorophyll reflection: G > B, G close to or > R, Hue 18-110):
    green_foliage = (
        (hue >= 18) & (hue <= 110) &
        (sat >= 20) & (val >= 20) &
        (g > b * 1.02) &
        ((g >= r * 0.85) | (2 * g - r - b > 5))
    )

    # 2. Chlorotic foliar tissue (yellowish margins, senescence, leaf blade):
    yellow_foliage = (
        (hue >= 12) & (hue < 22) &
        (sat >= 25) & (val >= 30) &
        (g > b * 1.10) & (r > b * 1.10) &
        (g >= r * 0.65) & (2 * g - r - b > -10)
    )

    # 3. Necrotic foliar lesions (brown spot lesions, blast spindle centers, hispa scratches):
    lesion_tissue = (
        (hue >= 5) & (hue <= 24) &
        (sat >= 15) & (val >= 15) & (val <= 210) &
        (r > b * 1.05) & (g > b * 1.01) &
        (r >= g * 0.85)
    )

    total_foliar = green_foliage | yellow_foliage | lesion_tissue
    green_ratio = float(green_foliage.mean())
    yellow_ratio = float(yellow_foliage.mean())
    lesion_ratio = float(lesion_tissue.mean())
    veg_ratio = green_ratio + yellow_ratio
    total_foliar_ratio = float(total_foliar.mean())

    # Decision 1: Minimum foliar / vegetative presence
    # Slender paddy leaf blades in 224x224 typically occupy 2.5% to 25% of image area
    if veg_ratio < 0.02 and total_foliar_ratio < 0.025:
        return False, "Invalid image: Please upload a paddy leaf image for disease detection.", {
            "error_type": "low_vegetation",
            "veg_ratio": round(veg_ratio * 100, 1),
            "green_ratio": round(green_ratio * 100, 1),
            "total_foliar": round(total_foliar_ratio * 100, 1)
        }

    # Decision 2: Lesion without foliage (e.g. human skin, leather, wood, brown fur)
    if lesion_ratio > 0.35 and green_ratio < 0.03:
        return False, "Invalid image: Please upload a paddy leaf image for disease detection.", {
            "error_type": "lesion_without_foliage",
            "lesion_ratio": round(lesion_ratio * 100, 1),
            "green_ratio": round(green_ratio * 100, 1)
        }

    # Decision 3: Dominance of non-plant colors (bright sky, blue ocean, red vehicle/clothing)
    # Bright blue objects / sky (val > 100, b > 90) without foliar tissue
    bright_blue = (b > g * 1.25) & (b > r * 1.25) & (sat > 40) & (val > 100)
    if float(bright_blue.mean()) > 0.30 and total_foliar_ratio < 0.03:
        return False, "Invalid image: Please upload a paddy leaf image for disease detection.", {
            "error_type": "blue_dominant",
            "blue_ratio": round(float(bright_blue.mean()) * 100, 1)
        }

    bright_red = (r > g * 1.30) & (r > b * 1.30) & (sat > 40) & (val > 60)
    if float(bright_red.mean()) > 0.25 and green_ratio < 0.03:
        return False, "Invalid image: Please upload a paddy leaf image for disease detection.", {
            "error_type": "red_dominant",
            "red_ratio": round(float(bright_red.mean()) * 100, 1)
        }

    # Decision 4: Foliar pixel vegetative chlorophyll / chromaticity check
    foliar_exg = (2 * g - r - b)[green_foliage]
    if len(foliar_exg) > 0 and float(foliar_exg.mean()) < 3.0:
        return False, "Invalid image: Please upload a paddy leaf image for disease detection.", {
            "error_type": "low_foliar_exg",
            "exg_mean": round(float(foliar_exg.mean()), 1)
        }

    # Decision 5: Paddy Leaf subject presence (center or central band)
    center_foliar = total_foliar[40:184, 40:184]
    if float(center_foliar.mean()) < 0.02 and total_foliar_ratio < 0.03:
        return False, "Invalid image: Please upload a paddy leaf image for disease detection.", {
            "error_type": "center_not_foliar",
            "center_foliar": round(float(center_foliar.mean()) * 100, 1)
        }

    return True, "Valid paddy leaf image.", {
        "veg_ratio": round(veg_ratio * 100, 1),
        "green_ratio": round(green_ratio * 100, 1),
        "lesion_ratio": round(lesion_ratio * 100, 1),
        "center_foliar": round(float(center_foliar.mean()) * 100, 1),
        "exg_mean": round(float(foliar_exg.mean()), 1) if len(foliar_exg) > 0 else 0.0
    }


def load_and_preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """Loads image bytes, converts to RGB, and transforms to tensor."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0)  # [1, 3, 256, 256]
    return tensor


def load_image_and_tensor(file_bytes: bytes) -> Tuple[torch.Tensor, Image.Image]:
    """Loads image bytes, converts to RGB, and returns both evaluation tensor and PIL Image."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0)  # [1, 3, 256, 256]
    return tensor, image


def extract_botanical_signals(image_pil: Image.Image) -> Tuple[float, float]:
    """
    Extracts foliar lesion and textural characteristics:
    1. scrape_ratio: Proportion of leaf tissue displaying pale, bleached/translucent longitudinal
       feeding scrapes characteristic of Paddy Hispa (Dicladispa armigera).
    2. brown_ratio: Proportion of leaf tissue displaying discrete dark reddish-brown necrotic spots
       characteristic of Brown Spot (Bipolaris oryzae).
    """
    im_scaled = image_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(im_scaled, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    im_hsv = np.array(im_scaled.convert("HSV"), dtype=np.float32)
    hue, sat, val = im_hsv[:, :, 0], im_hsv[:, :, 1], im_hsv[:, :, 2]

    # Leaf surface mask (excluding background)
    leaf = (val > 35) & ((g > b * 0.95) | (r > b * 0.95) | (g > 45))
    n_leaf = max(int(np.count_nonzero(leaf)), 1)

    # 1. Silvery/translucent feeding scrapes (Paddy Hispa)
    hispa_scrapes = leaf & (val > 120) & (sat < 65) & (np.abs(r - g) < 20) & (np.abs(g - b) < 25)
    scrape_ratio = float(np.count_nonzero(hispa_scrapes)) / n_leaf

    # 2. Distinct reddish-brown necrotic spots (Brown Spot)
    brown_spots = leaf & (r > g * 1.18) & (r > b * 1.30) & (r > 60) & (r < 185) & (sat > 50)
    brown_ratio = float(np.count_nonzero(brown_spots)) / n_leaf

    return scrape_ratio, brown_ratio


def calibrate_uncertain_logits(
    logits_in: np.ndarray,
    top_pil: Image.Image,
    bottom_pil: Optional[Image.Image] = None,
    view_type: str = "front"
) -> np.ndarray:
    """
    Calibrates logits only in the uncertainty regime when Brown Spot and Hispa predictions are ambiguous
    (gap < 2.5 logits and top logit < 3.2). Preserves confident model predictions without interference.
    """
    logits = np.copy(logits_in)
    bs_idx = 0  # Brown spot
    hp_idx = 1  # Healthy Paddy
    h_idx  = 2  # Hispa
    lb_idx = 3  # Leaf Blast

    sc_t, br_t = extract_botanical_signals(top_pil)
    if bottom_pil is not None:
        sc_b, br_b = extract_botanical_signals(bottom_pil)
        sc = max(sc_t, sc_b)
        br = max(br_t, br_b)
    else:
        sc, br = sc_t, br_t

    top_logit = float(np.max(logits))
    margin_bs_h = float(logits[bs_idx] - logits[h_idx])

    # Only calibrate if model is in the uncertainty regime between Brown Spot and Hispa
    if top_logit < 3.2 and abs(margin_bs_h) < 2.5:
        if (sc > 0.05 and br < 0.005) or (sc > 0.10 and sc > br * 3.0):
            # Longitudinal epidermal scrapes without necrotic circular spots -> Hispa
            boost = 1.6 + 2.0 * min(sc / 0.15, 1.5)
            logits[h_idx] += boost
            logits[bs_idx] -= boost * 0.4
        elif br > 0.008 and sc < 0.08:
            # Discrete dark necrotic lesions -> Brown spot
            boost = 1.5 + 2.0 * min(br / 0.015, 1.5)
            logits[bs_idx] += boost
            logits[h_idx] -= boost * 0.4
        elif view_type == "back" and abs(margin_bs_h) < 1.0:
            if sc > 0.03:
                logits[h_idx] += 1.2

    return logits


def run_inference(
    tensor_top: torch.Tensor,
    tensor_bottom: Optional[torch.Tensor] = None,
    image_top: Optional[Image.Image] = None,
    image_bottom: Optional[Image.Image] = None,
    view_type: str = "front"
) -> Dict[str, Any]:
    """Runs model inference, applies TTA & botanical calibration, and formats output probabilities and metadata."""
    if model is None:
        raise RuntimeError("Model is not initialized or checkpoint failed to load.")

    is_dual_view = tensor_bottom is not None and tensor_bottom is not tensor_top

    tensor_top = tensor_top.to(device)
    if tensor_bottom is not None:
        tensor_bottom = tensor_bottom.to(device)
    else:
        tensor_bottom = tensor_top  # Siamese fallback

    start_t = time.perf_counter()
    with torch.inference_mode():
        if is_dual_view:
            # Dual-View: ensemble canonical orientation (0.80) with symmetric inverted (0.20)
            logits_fwd = model(tensor_top, tensor_bottom)
            logits_rev = model(tensor_bottom, tensor_top)
            logits_tensor = 0.80 * logits_fwd + 0.20 * logits_rev
        else:
            # Single-View: Siamese feature extraction with horizontal flip TTA
            out1 = model(tensor_top, tensor_bottom)
            t_top_flip = torch.flip(tensor_top, dims=[3])
            t_bot_flip = torch.flip(tensor_bottom, dims=[3])
            out2 = model(t_top_flip, t_bot_flip)
            logits_tensor = (out1 + out2) / 2.0

        logits = logits_tensor[0].cpu().numpy()

    # Apply foliar calibration in the uncertainty regime if PIL image is available
    if image_top is not None:
        logits = calibrate_uncertain_logits(logits, image_top, bottom_pil=image_bottom, view_type=view_type)

    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    pred_idx = int(probs.argmax())
    pred_class = CLASSES[pred_idx]
    confidence_pct = round(float(probs[pred_idx]) * 100, 2)

    class_probabilities = {
        cls: round(float(probs[i]) * 100, 2)
        for i, cls in enumerate(CLASSES)
    }

    meta = DISEASE_METADATA.get(pred_class, {})

    return {
        "predicted_class": pred_class,
        "confidence": confidence_pct,
        "class_index": pred_idx,
        "probabilities": class_probabilities,
        "inference_time_ms": elapsed_ms,
        "disease_info": meta
    }


def extract_pair_id(filename: str) -> Optional[str]:
    """
    Extracts leaf specimen / pair identifier from filename according to dataset conventions:
    Examples:
      - 000206_t698.JPG -> 698
      - 000205_b698.JPG -> 698
      - leaf_12_T1.jpg -> 12
      - leaf_12_D1.jpg -> 12
      - P14_front.png -> P14
    """
    stem = Path(filename).stem.lower()

    # Pattern 1: suffix _t123 or _b123 or -d123
    m = re.search(r"[_-](?:t|b|d)(\d+)$", stem)
    if m:
        return f"Leaf-{m.group(1)}"

    # Pattern 2: trailing t123 or b123
    m = re.search(r"(?:t|b|d)(\d+)$", stem)
    if m:
        return f"Leaf-{m.group(1)}"

    # Pattern 3: sample prefix before _t/_b (e.g. leaf01_t1)
    m = re.search(r"^(.*?)[_-](?:t1?|b1?|d1?|front|back|top|bottom)", stem)
    if m and m.group(1):
        return m.group(1).capitalize()

    # Fallback to pure stem without view indicator
    cleaned = re.sub(r"[_-]?(t1?|b1?|d1?|top|bottom|front|back|adaxial|abaxial)", "", stem)
    return cleaned if cleaned else stem


def detect_leaf_view(filename: str) -> str:
    """
    Determines whether an image is:
    - 'front' (T1 / Top / Adaxial surface)
    - 'back' (D1 / B1 / Bottom / Abaxial surface)
    - 'unknown'
    """
    stem = Path(filename).stem.lower()
    if re.search(r"([_-]t\d*$|_t$|front|top|adaxial|^t\d*[_-])", stem):
        return "front"
    if re.search(r"([_-][bd]\d*$|_[bd]$|back|bottom|abaxial|^[bd]\d*[_-])", stem):
        return "back"
    return "unknown"


# ==============================================================================
# 4. FLASK APPLICATION & ROUTING
# ==============================================================================

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


@app.route("/")
def serve_index():
    """Serves the frontend dashboard."""
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
        return send_from_directory(str(FRONTEND_DIR), "index.html")
    return jsonify({
        "message": "Paddy Leaf Disease Detection Backend API is running.",
        "docs": "Endpoints available at /api/health, /api/model-info, /api/predict/front-view, /api/predict/dual-view, /api/predict/batch"
    })


@app.route("/<path:path>")
def serve_static(path):
    """Serves static assets from frontend folder."""
    if (FRONTEND_DIR / path).exists():
        return send_from_directory(str(FRONTEND_DIR), path)
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if model_load_status["loaded"] else "degraded",
        "model_loaded": model_load_status["loaded"],
        "device": str(device),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": time.time()
    })


@app.route("/api/network-info", methods=["GET"])
def api_network_info():
    """Returns local network IP addresses for easy mobile pairing."""
    ips = []
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        primary_ip = s.getsockname()[0]
        s.close()
        if primary_ip and not primary_ip.startswith("127."):
            ips.append(primary_ip)
    except Exception:
        pass

    try:
        import socket
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass

    port = int(os.environ.get("PORT", 5000))
    primary = ips[0] if ips else "127.0.0.1"
    return jsonify({
        "status": "success",
        "primary_ip": primary,
        "port": port,
        "all_ips": ips,
        "mobile_url": f"http://{primary}:{port}",
        "mobile_urls": [f"http://{ip}:{port}" for ip in ips],
        "local_url": f"http://127.0.0.1:{port}"
    })


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Returns technical details about the loaded model and dataset architecture."""
    return jsonify({
        "status": "success",
        "model_name": "Dual-View Siamese EfficientNet-B0 + cGAN Augmented Pipeline",
        "gan_augmentation": "Conditional Generative Adversarial Network (cGAN)",
        "checkpoint": model_load_status.get("checkpoint"),
        "loaded": model_load_status.get("loaded"),
        "error": model_load_status.get("error"),
        "device": model_load_status.get("device"),
        "device_name": model_load_status.get("device_name"),
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "total_parameters": model_load_status.get("total_parameters"),
        "input_resolution": f"{IMG_SIZE}x{IMG_SIZE} (RGB)",
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        },
        "feature_fusion": "Dual 1280-dim concatenated feature vectors (2560-dim)",
        "front_only_support": True,
        "front_only_mechanism": "Trained Siamese single-view projection (img_bottom = img_top fallback from author's training code)",
        "disease_reference": DISEASE_METADATA
    })


@app.route("/api/validate-image", methods=["POST"])
def api_validate_image():
    """
    Direct endpoint to verify whether an uploaded image contains an authentic paddy leaf.
    Does not run model inference.
    """
    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({
            "status": "error",
            "is_valid": False,
            "message": "No image provided for validation."
        }), 400

    filename = file.filename
    try:
        file_bytes = file.read()
        is_valid, message, details = verify_paddy_leaf_image(file_bytes)
        return jsonify({
            "status": "success",
            "is_valid": is_valid,
            "filename": filename,
            "message": message,
            "details": details
        })
    except Exception as e:
        logger.error(f"Image validation exception for {filename}: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "is_valid": False,
            "filename": filename,
            "message": f"Corrupted or invalid image: Failed to process image ({str(e)})"
        }), 500


@app.route("/api/predict/front-view", methods=["POST"])
def predict_front_view():
    """
    Analyzes exactly one front-view (T1) leaf image.
    1. First verifies whether the image contains a valid paddy leaf.
    2. If not a paddy leaf, returns an explicit warning and does NOT perform analysis.
    3. If valid, runs author's verified Siamese feature extraction fallback (tensor_bottom = tensor_top).
    """
    if not model_load_status["loaded"]:
        return jsonify({"status": "error", "message": f"Model not loaded: {model_load_status.get('error')}"}), 503

    # Accept single image under 'image' or 'images'
    file = request.files.get("image") or request.files.get("images")
    if not file:
        files = request.files.getlist("images")
        if files:
            file = files[0]

    if not file or file.filename == "":
        return jsonify({
            "status": "error",
            "error_type": "missing_image",
            "message": "No front-view image provided in request."
        }), 400

    filename = file.filename
    try:
        file_bytes = file.read()

        # Step 1: Verify whether the image contains a valid paddy leaf
        is_valid, val_msg, val_details = verify_paddy_leaf_image(file_bytes)
        if not is_valid:
            logger.warning(f"Front-view image verification rejected for '{filename}': {val_msg} ({val_details})")
            return jsonify({
                "status": "error",
                "error_type": val_details.get("error_type", "invalid_image"),
                "message": val_msg,
                "filename": filename,
                "validation_details": val_details
            }), 400

        # Step 2: Valid paddy leaf -> proceed with feature extraction and disease diagnosis
        tensor_top, pil_top = load_image_and_tensor(file_bytes)
        view_type = detect_leaf_view(filename)
        # Siamese single-view evaluation with TTA and foliar uncertainty calibration
        inf = run_inference(tensor_top, tensor_bottom=None, image_top=pil_top, view_type=view_type)

        result_data = {
            "status": "success",
            "filename": filename,
            "view": "T1 (Front View)",
            "inference_mode": "Front-View Siamese Fallback (T1)",
            "sample_id": extract_pair_id(filename),
            "predicted_class": inf["predicted_class"],
            "confidence": inf["confidence"],
            "class_index": inf["class_index"],
            "probabilities": inf["probabilities"],
            "inference_time_ms": inf["inference_time_ms"],
            "disease_info": inf["disease_info"],
            "validation_details": val_details,
            "predictions": [{
                "status": "success",
                "filename": filename,
                "view": "T1 (Front View)",
                "inference_mode": "Front-View Siamese Fallback (T1)",
                "sample_id": extract_pair_id(filename),
                **inf
            }]
        }
        return jsonify(result_data)
    except Exception as e:
        logger.error(f"Error processing front-view image {filename}: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error_type": "processing_error",
            "filename": filename,
            "message": f"Corrupted or invalid image: The uploaded file cannot be processed ({str(e)})."
        }), 500


@app.route("/api/predict/dual-view", methods=["POST"])
def predict_dual_view():
    """
    Dual-View Analysis: Requires both T1 (Front View) and D1 (Back View) of the leaf.
    1. First verifies whether both images contain valid paddy leaf surfaces.
    2. If either image is not a paddy leaf, displays an explicit message and blocks analysis.
    3. Runs exact dual-view feature fusion (1280 + 1280 = 2560 dims).
    """
    if not model_load_status["loaded"]:
        return jsonify({"status": "error", "message": f"Model not loaded: {model_load_status.get('error')}"}), 503

    if "front_image" not in request.files or "back_image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "Both 'front_image' (T1) and 'back_image' (D1) are required for dual-view analysis."
        }), 400

    front_file = request.files["front_image"]
    back_file = request.files["back_image"]

    if front_file.filename == "" or back_file.filename == "":
        return jsonify({
            "status": "error",
            "message": "One or both selected files are empty."
        }), 400

    try:
        front_bytes = front_file.read()
        back_bytes = back_file.read()

        # Step 1: Verify whether front-view image is an authentic paddy leaf
        is_v_f, msg_f, details_f = verify_paddy_leaf_image(front_bytes)
        if not is_v_f:
            logger.warning(f"Dual-view front image verification rejected for '{front_file.filename}': {msg_f}")
            display_msg = "Invalid image (Front View): Please upload a paddy leaf image for disease detection." if "paddy leaf" in msg_f else f"Front View: {msg_f}"
            return jsonify({
                "status": "error",
                "error_type": details_f.get("error_type", "invalid_image"),
                "message": display_msg,
                "failed_view": "front",
                "front_filename": front_file.filename,
                "back_filename": back_file.filename,
                "validation_details": details_f
            }), 400

        # Step 2: Verify whether back-view image is an authentic paddy leaf
        is_v_b, msg_b, details_b = verify_paddy_leaf_image(back_bytes)
        if not is_v_b:
            logger.warning(f"Dual-view back image verification rejected for '{back_file.filename}': {msg_b}")
            display_msg = "Invalid image (Back View): Please upload a paddy leaf image for disease detection." if "paddy leaf" in msg_b else f"Back View: {msg_b}"
            return jsonify({
                "status": "error",
                "error_type": details_b.get("error_type", "invalid_image"),
                "message": display_msg,
                "failed_view": "back",
                "front_filename": front_file.filename,
                "back_filename": back_file.filename,
                "validation_details": details_b
            }), 400

        # Step 3: Both views verified -> execute full Dual-View Siamese Fusion
        tensor_top, pil_top = load_image_and_tensor(front_bytes)
        tensor_bottom, pil_bottom = load_image_and_tensor(back_bytes)

        # Auto-align inverted views if user uploaded bottom view into front slot and top view into back slot
        v_f = detect_leaf_view(front_file.filename)
        v_b = detect_leaf_view(back_file.filename)
        if v_f == "back" and v_b == "front":
            tensor_top, tensor_bottom = tensor_bottom, tensor_top
            pil_top, pil_bottom = pil_bottom, pil_top
            logger.info(f"Auto-aligned inverted dual views: '{front_file.filename}' -> bottom, '{back_file.filename}' -> top")

        inf = run_inference(tensor_top, tensor_bottom, image_top=pil_top, image_bottom=pil_bottom, view_type="front")

        return jsonify({
            "status": "success",
            "inference_mode": "Dual-View Fusion (T1 + D1)",
            "front_filename": front_file.filename,
            "back_filename": back_file.filename,
            "sample_id": extract_pair_id(front_file.filename),
            "front_validation": details_f,
            "back_validation": details_b,
            **inf
        })
    except Exception as e:
        logger.error(f"Dual-view prediction failed: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error_type": "processing_error",
            "message": f"Corrupted or invalid image: Prediction failed ({str(e)})."
        }), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """
    Folder-Based Batch Analysis (High-Performance Parallel & Batched Pipeline):
    - Multi-threaded concurrent image decoding and resizing using ThreadPoolExecutor.
    - Batched PyTorch tensor evaluation for accelerated CPU vectorization.
    - Automatic T1/D1 pairing and Siamese fallback handling.
    """
    if not model_load_status["loaded"]:
        return jsonify({"status": "error", "message": f"Model not loaded: {model_load_status.get('error')}"}), 503

    files = request.files.getlist("files")
    if not files:
        return jsonify({"status": "error", "message": "No files received for batch analysis."}), 400

    mode = request.form.get("mode", "auto").lower()
    if mode not in ("auto", "dual", "front"):
        mode = "auto"

    start_total_t = time.perf_counter()

    # Step 1: Read raw file bytes
    file_bytes_map = {}
    for f in files:
        if f.filename:
            file_bytes_map[f.filename] = f.read()

    # Step 2: Concurrently preprocess and validate all images across CPU cores
    def _preprocess_single(item):
        fn, raw_bytes = item
        try:
            # Step 2a: Verify whether image is an authentic paddy leaf
            is_valid, msg, _ = verify_paddy_leaf_image(raw_bytes)
            if not is_valid:
                return fn, None, None, msg
            im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            tensor = eval_transform(im).unsqueeze(0)  # [1, 3, 256, 256]
            return fn, tensor, im, None
        except Exception as e:
            return fn, None, None, f"Corrupted or invalid image: Failed to process image ({str(e)})"

    num_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        preprocessed_items = list(pool.map(_preprocess_single, file_bytes_map.items()))

    tensor_map = {fn: tensor for fn, tensor, im, err in preprocessed_items if tensor is not None}
    image_map = {fn: im for fn, tensor, im, err in preprocessed_items if im is not None}
    error_map = {fn: err for fn, tensor, im, err in preprocessed_items if err is not None}

    batch_results = []
    class_distribution = {cls: 0 for cls in CLASSES}
    total_samples = 0
    healthy_count = 0
    diseased_count = 0
    unmatched_count = 0
    error_count = 0

    # Queue of samples ready for batched model evaluation
    inference_queue = []

    # Step 3 & 4: Process samples based on analysis mode
    if mode == "front":
        # Mode: Top-Only (T1) Evaluation
        # Evaluate each top/front view image individually without requiring or showing D1
        front_fns = [fn for fn in file_bytes_map.keys() if detect_leaf_view(fn) == "front"]
        if not front_fns:
            # If no filenames explicitly tagged as front, include all non-back images
            front_fns = [fn for fn in file_bytes_map.keys() if detect_leaf_view(fn) != "back"]
        if not front_fns:
            front_fns = list(file_bytes_map.keys())

        for front_fn in sorted(front_fns):
            total_samples += 1
            sample_id = extract_pair_id(front_fn) or Path(front_fn).stem
            if front_fn in error_map:
                error_count += 1
                batch_results.append({
                    "sample_id": sample_id,
                    "status": "error",
                    "mode": "Top-Only (T1)",
                    "front_filename": front_fn,
                    "back_filename": None,
                    "error": error_map[front_fn]
                })
            else:
                t_f = tensor_map[front_fn]
                v_f = detect_leaf_view(front_fn)
                inference_queue.append({
                    "sample_id": sample_id,
                    "mode": "Top-Only (T1)",
                    "front_filename": front_fn,
                    "back_filename": None,
                    "matched": True,
                    "t_top": t_f,
                    "t_bot": t_f,
                    "im_top": image_map[front_fn],
                    "im_bot": None,
                    "is_dual": False,
                    "view_type": v_f
                })
    else:
        # Step 3: Map filenames into logical sample groups for dual/auto pairing
        sample_groups = defaultdict(lambda: {"front": [], "back": [], "unknown": []})
        for filename in file_bytes_map.keys():
            view = detect_leaf_view(filename)
            sample_id = extract_pair_id(filename) or filename
            if view == "front":
                sample_groups[sample_id]["front"].append(filename)
            elif view == "back":
                sample_groups[sample_id]["back"].append(filename)
            else:
                sample_groups[sample_id]["unknown"].append(filename)

        # Step 4: Classify sample availability and prepare inference items
        for sample_id, views in sorted(sample_groups.items(), key=lambda x: str(x[0])):
            total_samples += 1
            front_list = views["front"]
            back_list = views["back"]
            unknown_list = views["unknown"]

            # Case A: Dual-view pair exists
            if front_list and back_list:
                front_fn = front_list[0]
                back_fn = back_list[0]

                if front_fn in error_map or back_fn in error_map:
                    error_count += 1
                    err_msg = error_map.get(front_fn) or error_map.get(back_fn)
                    batch_results.append({
                        "sample_id": sample_id,
                        "status": "error",
                        "mode": "Dual-View (T1 + D1)",
                        "front_filename": front_fn,
                        "back_filename": back_fn,
                        "error": err_msg
                    })
                else:
                    t_f = tensor_map[front_fn]
                    t_b = tensor_map[back_fn]
                    im_f = image_map[front_fn]
                    im_b = image_map[back_fn]

                    # Auto-align views if front is back-view and back is front-view
                    v_f = detect_leaf_view(front_fn)
                    v_b = detect_leaf_view(back_fn)
                    if v_f == "back" and v_b == "front":
                        t_top, t_bot = t_b, t_f
                        im_top, im_bot = im_b, im_f
                    else:
                        t_top, t_bot = t_f, t_b
                        im_top, im_bot = im_f, im_b

                    inference_queue.append({
                        "sample_id": sample_id,
                        "mode": "Dual-View (T1 + D1)",
                        "front_filename": front_fn,
                        "back_filename": back_fn,
                        "matched": True,
                        "t_top": t_top,
                        "t_bot": t_bot,
                        "im_top": im_top,
                        "im_bot": im_bot,
                        "is_dual": True,
                        "view_type": "front"
                    })

            # Case B: Front view only exists (Auto mode fallback to Top-Only)
            elif front_list and not back_list:
                front_fn = front_list[0]
                if mode == "auto":
                    if front_fn in error_map:
                        error_count += 1
                        batch_results.append({
                            "sample_id": sample_id,
                            "status": "error",
                            "mode": "Top-Only (T1)",
                            "front_filename": front_fn,
                            "back_filename": None,
                            "error": error_map[front_fn]
                        })
                    else:
                        t_f = tensor_map[front_fn]
                        inference_queue.append({
                            "sample_id": sample_id,
                            "mode": "Top-Only (T1)",
                            "front_filename": front_fn,
                            "back_filename": None,
                            "matched": True,
                            "t_top": t_f,
                            "t_bot": t_f,
                            "im_top": image_map[front_fn],
                            "im_bot": None,
                            "is_dual": False,
                            "view_type": "front"
                        })
                else:
                    unmatched_count += 1
                    batch_results.append({
                        "sample_id": sample_id,
                        "status": "unmatched",
                        "mode": "Dual-View Required",
                        "front_filename": front_fn,
                        "back_filename": None,
                        "matched": False,
                        "reason": "Dual-View mode requires both T1 and D1 views. Missing back image."
                    })

            # Case C: Back view only exists
            elif back_list and not front_list:
                back_fn = back_list[0]
                if mode == "auto":
                    # In Auto mode, allow single back-view specimen diagnosis fallback
                    if back_fn in error_map:
                        error_count += 1
                        batch_results.append({
                            "sample_id": sample_id,
                            "status": "error",
                            "mode": "Single-View (B1 Fallback)",
                            "front_filename": None,
                            "back_filename": back_fn,
                            "error": error_map[back_fn]
                        })
                    else:
                        t_b = tensor_map[back_fn]
                        inference_queue.append({
                            "sample_id": sample_id,
                            "mode": "Single-View (B1 Fallback)",
                            "front_filename": None,
                            "back_filename": back_fn,
                            "matched": True,
                            "t_top": t_b,
                            "t_bot": t_b,
                            "im_top": image_map[back_fn],
                            "im_bot": None,
                            "is_dual": False,
                            "view_type": "back"
                        })
                else:
                    unmatched_count += 1
                    batch_results.append({
                        "sample_id": sample_id,
                        "status": "unmatched",
                        "mode": "Dual-View Required",
                        "front_filename": "Missing Front View (T1)",
                        "back_filename": back_list[0],
                        "matched": False,
                        "reason": "Dual-View mode requires both T1 and D1 views. Missing front image."
                    })

            # Case D: Unknown naming convention
            elif unknown_list:
                for unk_fn in unknown_list:
                    if mode == "auto":
                        if unk_fn in error_map:
                            error_count += 1
                            batch_results.append({
                                "sample_id": sample_id,
                                "status": "error",
                                "mode": "Top-Only (T1)",
                                "front_filename": unk_fn,
                                "back_filename": None,
                                "error": error_map[unk_fn]
                            })
                        else:
                            t_unk = tensor_map[unk_fn]
                            v_unk = detect_leaf_view(unk_fn)
                            inference_queue.append({
                                "sample_id": sample_id,
                                "mode": "Top-Only (T1)",
                                "front_filename": unk_fn,
                                "back_filename": None,
                                "matched": True,
                                "t_top": t_unk,
                                "t_bot": t_unk,
                                "im_top": image_map[unk_fn],
                                "im_bot": None,
                                "is_dual": False,
                                "view_type": v_unk
                            })
                    else:
                        unmatched_count += 1
                        batch_results.append({
                            "sample_id": sample_id,
                            "status": "unmatched",
                            "mode": "Unmatched",
                            "front_filename": unk_fn,
                            "back_filename": None,
                            "matched": False,
                            "reason": "Filename does not specify T1 or D1 view tag."
                        })

    # Step 5: Execute calibrated model inference on queued samples
    for item in inference_queue:
        t_top = item["t_top"]
        t_bot = item["t_bot"]
        im_top = item.get("im_top")
        im_bot = item.get("im_bot")
        is_dual = item.get("is_dual", False)
        v_type = item.get("view_type", "front")

        # Run exact calibrated inference matching single-view and dual-view pipelines
        inf = run_inference(
            t_top,
            tensor_bottom=t_bot if is_dual else None,
            image_top=im_top,
            image_bottom=im_bot,
            view_type=v_type
        )

        pred_class = inf["predicted_class"]
        conf = inf["confidence"]
        pred_idx = inf["class_index"]
        class_probs = inf["probabilities"]
        meta = inf["disease_info"]

        class_distribution[pred_class] += 1
        if pred_class == "Healthy Paddy":
            healthy_count += 1
        else:
            diseased_count += 1

        batch_results.append({
            "sample_id": item["sample_id"],
            "status": "success",
            "mode": item["mode"],
            "front_filename": item["front_filename"],
            "back_filename": item["back_filename"],
            "matched": item["matched"],
            "predicted_class": pred_class,
            "confidence": conf,
            "class_index": pred_idx,
            "probabilities": class_probs,
            "inference_time_ms": inf["inference_time_ms"],
            "disease_info": meta
        })

    # Summary statistics
    analyzed_success = [r for r in batch_results if r.get("status") == "success"]
    avg_confidence = (
        round(sum(r["confidence"] for r in analyzed_success) / len(analyzed_success), 2)
        if analyzed_success else 0.0
    )
    total_elapsed_ms = round((time.perf_counter() - start_total_t) * 1000, 2)

    return jsonify({
        "status": "success",
        "summary": {
            "total_files": len(file_bytes_map),
            "total_samples": total_samples,
            "processed_count": len(analyzed_success),
            "unmatched_count": unmatched_count,
            "error_count": error_count,
            "healthy_count": healthy_count,
            "diseased_count": diseased_count,
            "average_confidence": avg_confidence,
            "class_distribution": class_distribution,
            "mode": mode,
            "total_processing_ms": total_elapsed_ms
        },
        "results": batch_results
    })


# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Bind to 0.0.0.0 to allow mobile phones and LAN devices to connect
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))

    # Detect local LAN IP address to display the exact URL for the phone
    lan_ip = "127.0.0.1"
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    logger.info(f"Starting Paddy Disease Detection Server on http://0.0.0.0:{port}")
    print("\n" + "=" * 66)
    print("[PADDYSCAN AI SERVER RUNNING - PC & MOBILE READY]")
    print(f"-> Local Access (On PC):     http://localhost:{port}")
    print(f"-> Mobile Access (On Phone): http://{lan_ip}:{port}")
    print("=" * 66)
    print("TO OPEN ON YOUR MOBILE PHONE:")
    print("  1. Connect phone to the same Wi-Fi or Mobile Hotspot / USB Tether.")
    print(f"  2. Open browser on your phone and go to: http://{lan_ip}:{port}")
    print("=" * 66 + "\n", flush=True)

    app.run(host=host, port=port, debug=False)
