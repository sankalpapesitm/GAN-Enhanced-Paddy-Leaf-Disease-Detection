# PaddyScan AI • Dual-View Paddy Leaf Disease Diagnostic System

A complete agricultural frontend and PyTorch machine learning integration system powered by an existing trained **Dual-View Siamese EfficientNet-B0** model (`best_dual_efficientnet_b0.pth`).

---

## 🌾 Project Overview

Paddy leaf diseases—such as **Leaf Blast**, **Brown Spot**, and **Hispa**—cause catastrophic yield losses across rice-producing regions. While classical computer vision models only observe a leaf's adaxial (top/front) surface, foliar pathogens frequently exhibit distinct morphological patterns on the abaxial (bottom/back) surface.

This project delivers a responsive, agricultural dashboard connected to a trained PyTorch Siamese model checkpoint without retraining, replacing, or simulating model inference.

### Key Highlights
- **Direct Checkpoint Integration**: Seamlessly loads `best_dual_efficientnet_b0.pth` (5,321,856 parameters).
- **Siamese 2560D Feature Fusion**: Extracts 1280 features from the Front View (T1) and 1280 features from the Back View (D1), fusing them into a 2560-dimensional representation.
- **Native Front-Only Support (T1)**: Implements the author's verified training fallback (`img_bottom = img_top`) for single adaxial surface diagnosis with transparent user indication.
- **Folder-Based Batch Processing**: Automatically identifies leaf pairs (`_t` and `_b`, `_t1` and `_d1`) by sample ID, dynamically manages pairing strategies, and exports results to CSV.

---

## 📁 Project Structure

```
PaddyDiseaseApp/
├── frontend/
│   ├── index.html              # Main responsive dashboard layout
│   ├── css/
│   │   └── style.css           # Modern agricultural design system (Emerald & Mint)
│   ├── js/
│   │   └── script.js           # Client-side validation, Chart.js analytics & API connector
│   └── assets/
│       └── samples/            # Ready-to-use sample leaf images for instant testing
│           ├── sample_01_t1.jpg
│           ├── sample_01_b1.jpg
│           ├── sample_02_t2.jpg
│           ├── sample_02_b2.jpg
│           ├── sample_03_t3.jpg
│           └── sample_04_b4.jpg
├── backend/
│   ├── app.py                  # Flask REST API & PyTorch model loader
│   ├── requirements.txt        # Backend dependencies
│   └── models/
│       └── best_dual_efficientnet_b0.pth # Trained PyTorch checkpoint
└── README.md                   # Full documentation & run instructions
```

---

## 🎯 Supported Disease Classes

The model classifies leaves into 4 official classes:

| Index | Class Name | Severity | Pathogen / Cause |
|---|---|---|---|
| **0** | `Brown spot` | Moderate to High | *Bipolaris oryzae* (*Cochliobolus miyabeanus*) |
| **1** | `Healthy Paddy` | None (Normal) | Physiologically balanced leaf |
| **2** | `Hispa` | Moderate to Severe | *Dicladispa armigera* (Leaf miner beetle) |
| **3** | `Leaf Blast` | Critical / Severe | *Magnaporthe oryzae* (*Pyricularia oryzae*) |

---

## 🖥️ Three Functional Diagnostic Sections

### 1. Front-View Analysis (T1)
- Accepts **exactly one front-view paddy leaf image** (single-image input only).
- Real-time image preview with filename, file size, replace, and remove controls.
- Quick test sample loader button for instant single-leaf evaluation.
- Evaluates the image through the author's verified Siamese feature extraction fallback.
- Displays the **single highest-accuracy predicted disease**, confidence percentage, 4-class softmax probability distribution, and agronomic management advice.
- Interactive CSV download for the single prediction result.

### 2. Dual-View Analysis (T1 + D1)
- Split upload cards for **Front View (T1 / Adaxial)** and **Back View (D1 / Abaxial)**.
- Visual readiness indicator: ensures both surfaces of the leaf are uploaded before allowing analysis.
- Executes full dual-stream Siamese feature fusion (2560 dimensions).
- Displays synchronized side-by-side leaf comparison, pathogen identification, confidence gauge, and latency metrics.

### 3. Folder-Based Batch Analysis
- Allows uploading entire dataset folders (`webkitdirectory`) or multiple files.
- Automatically detects T1 vs D1 views using dataset naming conventions:
  - Patterns: `[ID]_t[N]`, `[ID]_b[N]`, `leaf_[ID]_T1`, `leaf_[ID]_D1`, `front`, `back`, etc.
- **Three Strategic Modes**:
  1. **Auto-Detect & Diagnose**: Dual-view for matched pairs; Front-Only fallback for solitary T1; flags solitary D1 as unmatched without guessing.
  2. **Strict Dual-View Only**: Requires both T1 and D1; marks solitary views as unmatched.
  3. **Front-Only (T1) Batch**: Evaluates all adaxial images via front fallback; ignores back views.
- Real-time progress bar tracking actual batch progress.
- Interactive **Chart.js** disease distribution doughnut chart.
- Searchable & filterable results table with Sample ID, file pairing, confidence, status badges, and sample detail modal.
- Export entire batch to CSV.

---

## 🔌 Flask Backend API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check, device status (CPU/GPU), and model load state. |
| `GET` | `/api/model-info` | Model architecture specifications, classes, parameter count, and disease metadata. |
| `POST` | `/api/predict/front-view` | Accepts one or more front-view images (`images` field). Returns list of predictions. |
| `POST` | `/api/predict/dual-view` | Accepts `front_image` and `back_image`. Executes 2560D dual-stream inference. |
| `POST` | `/api/predict/batch` | Accepts multi-file upload (`files`) and `mode` (`auto`, `dual`, `front`). Returns paired analysis, summary statistics, and distribution counts. |

---

## 🚀 Setup & Run Instructions (Windows)

### Prerequisites
- Python 3.9+ (Python 3.10, 3.11, 3.12, 3.14 tested)
- A modern web browser (Google Chrome, Microsoft Edge, or Mozilla Firefox)

### Step 1: Open PowerShell or Command Prompt
Navigate to the project root directory:
```powershell
cd "c:\Users\sanka\OneDrive\Desktop\gan based paddy leaf\PaddyDiseaseApp"
```

### Step 2: Install Python Dependencies
```powershell
pip install -r backend/requirements.txt
```

### Step 3: Start the Flask Backend
```powershell
python backend/app.py
```
You will see output indicating that the model weights matched and loaded:
```
[INFO] Loading checkpoint from: ...\backend\models\best_dual_efficientnet_b0.pth
[INFO] Model state_dict loaded successfully: <All keys matched successfully>
[INFO] Model ready on device: cpu (CPU)
Starting Paddy Disease Detection Server on http://127.0.0.1:5000
```

### Step 4: Access the Frontend
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```
*Note: The Flask application directly serves the frontend dashboard, or you can open `frontend/index.html` directly in your browser—both methods are fully supported!*

### Step 5: Test with Demo Samples
- In **Front-View Analysis**, click **"Load Test Sample (T1)"** &rarr; **"Analyze Front View"**.
- In **Dual-View Analysis**, click **"Load Test Pair (T1 + D1)"** &rarr; **"Analyze Both Views"**.
- In **Folder Batch Analysis**, click **"Load Demo Batch Samples"** &rarr; **"Run Batch Analysis"**.
