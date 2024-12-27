import cv2
import numpy as np
from ultralytics import YOLO
import logging
from pathlib import Path
from typing import Tuple, List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShelfDetector:
    """
    A class to handle shelf product detection using YOLO and OpenCV.
    """
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the ShelfDetector with a YOLO model and configuration.
        
        Args:
            model_path: Path to the YOLO model weights
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        try:
            self.model = YOLO(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better detection results.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB for YOLO
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Apply slight Gaussian blur to reduce noise
        image_blur = cv2.GaussianBlur(image_normalized, (3, 3), 0)
        
        return image_blur

    def detect_products(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect products on shelves in the given image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple containing:
                - Annotated image with bounding boxes
                - List of detection results with coordinates and confidence scores
        """
        # Read and validate image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from {image_path}")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise

        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Perform detection
        try:
            results = self.model(processed_image)[0]
            detections = []
            
            # Process each detection
            for box in results.boxes:
                confidence = float(box.conf)
                
                if confidence >= self.confidence_threshold:
                    # Get coordinates and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class': class_name
                    }
                    detections.append(detection)
                    
                    # Draw bounding box on image
                    cv2.rectangle(image, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 
                                2)
                    
                    # Add label
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(image, 
                              label, 
                              (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              (0, 255, 0),
                              2)
            
            logger.info(f"Detected {len(detections)} products")
            return image, detections
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate the usage of ShelfDetector.
    """
    # Initialize detector
    detector = ShelfDetector()
    
    # Process sample images from a directory
    image_dir = Path("sample_images")
    output_dir = Path("output_images")
    output_dir.mkdir(exist_ok=True)
    
    for image_path in image_dir.glob("*.jpg"):
        try:
            # Perform detection
            annotated_image, detections = detector.detect_products(str(image_path))
            
            # Save results
            output_path = output_dir / f"detected_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_image)
            
            # Print detection results
            logger.info(f"Results for {image_path.name}:")
            for i, detection in enumerate(detections, 1):
                logger.info(f"Detection {i}: {detection}")
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()