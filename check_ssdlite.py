import cv2
import time
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from PIL import Image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained weights and model
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    model.to(device)
    model.eval()
    
    # Instantiate the transform pipeline from the weights
    transform = weights.transforms()  # expects only the image
    categories = weights.meta["categories"]
    
    
    print("Starting Webcam")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    

    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Unable to grab frame.")
            break

        # Convert the frame from BGR (OpenCV) to RGB (PIL)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Apply the transform to get a 3D tensor [C, H, W]
        image_tensor = transform(pil_image).to(device)
        
        # Pass the tensor inside a list (model expects a list of images)
        with torch.no_grad():
            outputs = model([image_tensor])
        
        output = outputs[0]
        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        
        threshold = 0.5  # Confidence threshold
        for box, label, score in zip(boxes, labels, scores):
            if score < threshold:
                continue
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{categories[label]}: {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("SSDLite Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
