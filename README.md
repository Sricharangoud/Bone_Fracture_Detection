# OsteoScan — Bone Fracture Detection

AI-powered web app that classifies X-ray images as **Normal Bone** or **Fractured Bone**.

## Folder Structure

```
bone_fracture_app/
├── app.py                  ← Flask backend + model inference
├── requirements.txt        ← Python dependencies
├── model/
│   └── bone_fracture_model.h5   ← ⚠ Place YOUR trained model here
├── templates/
│   └── index.html          ← Upload UI
└── static/
    └── style.css           ← Styles
```

## Setup & Run

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your model
Copy your trained TensorFlow/Keras model to:
```
model/bone_fracture_model.h5
```
> If no model file is found, the app runs in **demo mode** (random predictions).

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

## Model conventions

| Output shape | Expected classes        |
|--------------|-------------------------|
| `(1, 1)`     | Sigmoid — 0=Normal, 1=Fractured |
| `(1, 2+)`    | Softmax — index 0=Normal, index 1=Fractured |

Input images are automatically resized to **224×224** and normalized to `[0, 1]`.
Change `IMG_SIZE` and `MODEL_PATH` in `app.py` if your model differs.

## Notes
- For research & educational use only. Not a clinical diagnostic tool.
- Compatible with Python 3.9+.
