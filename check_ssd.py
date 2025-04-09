import cv2
import time
import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from PIL import Image

def main():
    # Use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained weights and model for SSD300 VGG16
    weights = SSD300_VGG16_Weights.DEFAULT
    model = ssd300_vgg16(weights=weights)
    model.to(device)
    model.eval()
    
    # Instantiate the transform pipeline from the weights
    transform = weights.transforms()  # expects only the image as input
    categories = weights.meta["categories"]

    print("Starting Webcam")
    # Use DirectShow backend for Windows
    # cap = cv2.VideoCapture("Video/random_youtube_video.mp4")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Unable to grab frame.")
            break

        # Convert the frame from BGR (OpenCV) to RGB (PIL) format
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Apply the transform to get a 3D tensor of shape [C, H, W] in the range [0, 1]
        image_tensor = transform(pil_image).to(device)
        
        # The model expects a list of tensors, one for each image
        with torch.no_grad():
            outputs = model([image_tensor])
        
        # Get the predictions for the first (and only) image
        output = outputs[0]
        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        
        # Set a confidence threshold
        threshold = 0.5
        for box, label, score in zip(boxes, labels, scores):
            if score < threshold:
                continue
            # Draw bounding box on the frame
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{categories[label]}: {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame with detection boxes
        cv2.imshow("SSD300 VGG16 Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
