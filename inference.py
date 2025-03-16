import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from PIL import Image
import signal
from contextlib import contextmanager
import threading

# Initialize FastAPI
app = FastAPI()

# Basic authentication setup
security = HTTPBasic()
VALID_USERNAME = "admin"
VALID_PASSWORD = "password123"  # Change this to a secure password

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != VALID_USERNAME or credentials.password != VALID_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# Load VGG16 and modify the classifier
vgg16 = models.vgg16()
vgg16.classifier[6] = torch.nn.Linear(4096, 10)  # Assuming 10 classes
vgg16.load_state_dict(torch.load("model_weight.pth", map_location=device))
vgg16.to(device)
vgg16.eval()

# Define the input schema
class ImageInput(BaseModel):
    pixels: list[float]  # List of 784 float pixel values

# Define the transformation pipeline
data_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor first
    transforms.Resize((224, 224), antialias=True),  # Resize to 224x224
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to RGB by repeating channels
])

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Prediction timed out")
    
    # Set the signal handler and a 5-second alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
import asyncio

@app.post("/predict")
async def predict(data: ImageInput, credentials: HTTPBasicCredentials = Depends(authenticate)):
    if len(data.pixels) != 784:
        raise HTTPException(status_code=400, detail="Invalid input: Expected 784 pixel values")

    try:
        img_array = np.array(data.pixels, dtype=np.uint8).reshape(28, 28)
        img_array = img_array / 255.0
        img = Image.fromarray(img_array)
        img = data_transform(img).unsqueeze(0).to(device)

        print(f"Input tensor shape: {img.shape}")

        async def run_model():
            with torch.no_grad():
                return vgg16(img)

        # Set timeout for model inference (10 seconds)
        try:
            output = await asyncio.wait_for(run_model(), timeout=100)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Model inference timed out")

        if output is None:
            raise ValueError("Model returned None output")

        predicted_class = torch.argmax(output, dim=1).item()
        confidence = float(torch.softmax(output, dim=1)[0][predicted_class])

        response = {
            "status": "success",
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.4f}"
        }
        return response

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
