import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time
import math

@dataclass
class DetectedObject:
    """Represents a detected object with its properties and AR overlay settings"""
    bbox: List[int]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    price: float = 0.0
    stock: int = 0
    highlight_color: Tuple[int, int, int] = (0, 255, 0)

class AREffect:
    """Manages animated visual effects for AR overlays"""
    def __init__(self):
        self.pulse_phase = 0
        self.last_update = time.time()
    
    def pulse_animation(self, intensity: float = 1.0) -> float:
        """Creates a pulsing animation effect"""
        current_time = time.time()
        self.pulse_phase += (current_time - self.last_update) * 3
        self.last_update = current_time
        
        # Create smooth pulse between 0 and 1
        pulse = (math.sin(self.pulse_phase) + 1) / 2
        return pulse * intensity

class ShelfAR:
    """Handles AR overlays for shelf products"""
    def __init__(self):
        self.effect_manager = AREffect()
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.base_font_scale = 0.6
        
    def create_info_panel(self, obj: DetectedObject) -> np.ndarray:
        """Creates an semi-transparent information panel for a product"""
        # Create panel background
        panel_width = 200
        panel_height = 100
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Add product information
        text_color = (255, 255, 255)
        cv2.putText(panel, f"Product: {obj.class_name}", 
                   (10, 25), self.font, 0.5, text_color, 1)
        cv2.putText(panel, f"Price: ${obj.price:.2f}", 
                   (10, 50), self.font, 0.5, text_color, 1)
        cv2.putText(panel, f"In Stock: {obj.stock}", 
                   (10, 75), self.font, 0.5, text_color, 1)
        
        return panel
    
    def add_floating_arrow(self, image: np.ndarray, 
                          target_point: Tuple[int, int], 
                          color: Tuple[int, int, int]):
        """Adds an animated floating arrow pointing to a product"""
        pulse = self.effect_manager.pulse_animation()
        
        # Arrow starts 50 pixels above target
        start_point = (target_point[0], target_point[1] - 50 - int(pulse * 10))
        
        # Draw arrow shaft
        cv2.line(image, start_point, 
                (start_point[0], target_point[1]), 
                color, 2)
        
        # Draw arrow head
        arrow_head_length = 10
        cv2.line(image, 
                (start_point[0], target_point[1]), 
                (start_point[0] - arrow_head_length, target_point[1] - arrow_head_length),
                color, 2)
        cv2.line(image, 
                (start_point[0], target_point[1]),
                (start_point[0] + arrow_head_length, target_point[1] - arrow_head_length),
                color, 2)

    def create_highlight_effect(self, image: np.ndarray, obj: DetectedObject):
        """Creates a pulsing highlight effect around detected objects"""
        x1, y1, x2, y2 = obj.bbox
        pulse = self.effect_manager.pulse_animation(0.5)
        
        # Create pulsing thickness for rectangle
        thickness = max(1, int(2 + pulse * 3))
        
        # Draw rectangle with rounded corners
        cv2.rectangle(image, (x1, y1), (x2, y2), 
                     obj.highlight_color, thickness)
        
        # Add subtle glow effect
        alpha = 0.3 * pulse
        overlay = image.copy()
        cv2.rectangle(overlay, (x1-5, y1-5), (x2+5, y2+5), 
                     obj.highlight_color, -1)
        cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)

    def add_overlays(self, image: np.ndarray, 
                    detected_objects: List[DetectedObject]) -> np.ndarray:
        """Adds all AR overlays to the image"""
        output = image.copy()
        
        for obj in detected_objects:
            # Add highlight effect
            self.create_highlight_effect(output, obj)
            
            # Calculate anchor points for overlays
            x1, y1, x2, y2 = obj.bbox
            center_x = (x1 + x2) // 2
            
            # Add floating arrow
            self.add_floating_arrow(output, (center_x, y1), obj.highlight_color)
            
            # Create and overlay info panel
            panel = self.create_info_panel(obj)
            panel_y = max(0, y1 - panel.shape[0] - 10)
            panel_x = min(image.shape[1] - panel.shape[1], center_x)
            
            # Create fade effect for panel
            alpha = 0.8 + self.effect_manager.pulse_animation(0.2)
            roi = output[panel_y:panel_y + panel.shape[0], 
                        panel_x:panel_x + panel.shape[1]]
            cv2.addWeighted(panel, alpha, roi, 1-alpha, 0, roi)
            
            # Add confidence score with dynamic color
            conf_color = (int(255 * (1-obj.confidence)), 
                         int(255 * obj.confidence), 0)
            conf_text = f"{obj.confidence*100:.1f}%"
            cv2.putText(output, conf_text, 
                       (x1, y1-5), self.font, 
                       self.base_font_scale, conf_color, 2)
        
        return output

def main():
    """Demo usage of the ShelfAR system"""
    # Initialize AR system
    ar_system = ShelfAR()
    
    # Create some sample detected objects
    sample_objects = [
        DetectedObject(
            bbox=[100, 100, 300, 400],
            class_name="Cereal Box",
            confidence=0.92,
            price=4.99,
            stock=15,
            highlight_color=(0, 255, 0)
        ),
        DetectedObject(
            bbox=[350, 150, 500, 380],
            class_name="Soda Bottle",
            confidence=0.88,
            price=2.49,
            stock=8,
            highlight_color=(255, 165, 0)
        )
    ]
    
    # Load and process a sample image
    image = cv2.imread("product_test.jpg")
    
    if image is not None:
        # Process frames in a loop to show animation
        while True:
            # Create AR overlay
            ar_image = ar_system.add_overlays(image, sample_objects)
            
            # Display result
            cv2.imshow("Shelf AR Demo", ar_image)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()