from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('/content/polution_best.pt')  # Adjust the path to your model file

app = FastAPI()

@app.post("/predict")
async def root():
    img_path = 'https://res.cloudinary.com/dlqknqol3/image/upload/v1730051702/kjf5uxesjxufzhpzaxa9.jpg'  # Change tthis to your image path
    img = io.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    confidence_threshold = 0.4 

    results = model(img, conf=confidence_threshold)

    predictions = results[0]
    return predictions
    



