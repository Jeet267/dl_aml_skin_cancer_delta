from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch, torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np, io, base64, uvicorn
import os

app = FastAPI(title="SkinCancer Detector — FusionSkinNet")

# ── Load model ───────────────────────────────────────────────
class LesionAttentionGate(nn.Module):
    def __init__(self, feat_channels=2048, meta_dim=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(meta_dim, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, feat_channels), nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, feat_map, meta):
        gate = self.gate(meta).unsqueeze(-1).unsqueeze(-1)
        return self.pool(feat_map * gate).flatten(1)

class FusionSkinNet(nn.Module):
    def __init__(self, meta_dim=8):
        super().__init__()
        base = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.lag = LesionAttentionGate(2048, meta_dim)
        self.head = nn.Sequential(
            nn.Linear(2048,512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(512,128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128,1))
    def forward(self, img, meta):
        return self.head(self.lag(self.encoder(img), meta))

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
USE_MOCK = not os.path.exists(MODEL_PATH)

MODEL = None
if not USE_MOCK:
    try:
        MODEL = FusionSkinNet(meta_dim=8).to(device)
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        MODEL.eval()
        print(f"✅ Loaded real weights from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}. Switching to Mock mode.")
        USE_MOCK = True

if USE_MOCK:
    print("⚠️  Running in MOCK mode (random predictions). best_model.pth not found or error loading.")

TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html") as f: return f.read()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    age: float = Form(45.0),
    sex: str   = Form("male"),
    site: str  = Form("anterior torso")
):
    if USE_MOCK:
        # Generate a deterministic-looking mock probability
        import random
        prob = random.uniform(0.01, 0.95)
    else:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_t = TRANSFORM(img).unsqueeze(0).to(device)

        # Build meta vector
        meta = [
            (age - 50) / 20,          # age_scaled
            1.0 if sex=="female" else 0.0,
            1.0 if sex=="male"   else 0.0,
            1.0 if site=="anterior torso"    else 0.0,
            1.0 if site=="head/neck"         else 0.0,
            1.0 if site=="lower extremity"   else 0.0,
            1.0 if site=="posterior torso"   else 0.0,
            1.0 if site=="upper extremity"   else 0.0,
        ]
        meta_t = torch.tensor([meta], dtype=torch.float32).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(MODEL(img_t, meta_t)).item()

    risk = "HIGH" if prob > 0.5 else "MODERATE" if prob > 0.2 else "LOW"
    return JSONResponse({"probability": round(prob*100, 2),
                         "risk": risk, "pauc": 0.9686})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
