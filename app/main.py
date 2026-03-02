import os
import sys
import io
import torch
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

# Ensure we can import from scripts
# Get the absolute path to the parent directory (d:\Code\Profiler)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
scripts_dir = os.path.join(parent_dir, "scripts")
sys.path.append(scripts_dir)

try:
    from model import FairFaceVGG16
except ImportError:
    # Fallback if running from root
    from scripts.model import FairFaceVGG16

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

# Templates
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# --- Model & Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHTS_PATH = os.path.join(parent_dir, "weights", "fairface_vgg16_weights.pth")
RACE_LABELS = {
    0: 'White', 1: 'Black', 2: 'Latino_Hispanic',
    3: 'East Asian', 4: 'Southeast Asian',
    5: 'Indian', 6: 'Middle Eastern'
}

model = None

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def load_model_app():
    global model
    if model is None:
        print(f"Loading weights from {WEIGHTS_PATH}...")
        model = FairFaceVGG16(num_races=len(RACE_LABELS))
        if os.path.exists(WEIGHTS_PATH):
            state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
        else:
            print("WARNING: Weights not found!")

# Load model on startup
load_model_app()
transform = get_inference_transform()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        gender_logit, age_pred, race_logits = model(input_tensor)
        
    # Process results
    gender_prob = gender_logit.item()
    gender = "Female" if gender_prob >= 0.5 else "Male"
    # Provide the probability of the predicted class
    display_prob = gender_prob if gender == "Female" else 1 - gender_prob
    
    age = round(age_pred.item(), 1)
    
    race_idx = torch.argmax(race_logits, dim=1).item()
    race = RACE_LABELS.get(race_idx, "Unknown")
    
    return {
        "gender": gender,
        "gender_prob": display_prob,
        "age": age,
        "race": race
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
