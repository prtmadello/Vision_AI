"""
Image processing utilities for Vision AI system.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image processing utilities for Vision AI."""
    
    def __init__(self):
        """Initialize image processor."""
        pass
    
    def draw_bounding_box(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        confidence: float = 0.0,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding box on image.
        
        Args:
            image: Input image
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            label: Label text
            confidence: Confidence score
            color: Box color (B, G, R)
            thickness: Line thickness
            
        Returns:
            Image with drawn bounding box
        """
        x1, y1, x2, y2 = bbox
        image_copy = image.copy()
        
        # Draw rectangle
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if label:
            label_text = f"{label}"
            if confidence > 0:
                label_text += f" ({confidence:.2f})"
            
            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )
            
            # Draw label background
            cv2.rectangle(
                image_copy,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image_copy,
                label_text,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        return image_copy
    
    def draw_multiple_boxes(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Draw multiple bounding boxes on image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            color_map: Optional color mapping for different classes
            
        Returns:
            Image with drawn bounding boxes
        """
        if color_map is None:
            color_map = {
                'face': (0, 255, 0),      # Green
                'person': (255, 0, 0),     # Blue
                'unknown': (0, 0, 255)     # Red
            }
        
        annotated = image.copy()
        
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            label = detection.get('label', 'unknown')
            confidence = detection.get('confidence', 0.0)
            
            # Get color for this class
            color = color_map.get(label, color_map['unknown'])
            
            # Draw bounding box
            annotated = self.draw_bounding_box(
                annotated, (x1, y1, x2, y2), label, confidence, color
            )
        
        return annotated
    
    def draw_tracking_boxes(
        self,
        image: np.ndarray,
        tracks: List[Dict[str, Any]],
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Draw tracking boxes with track IDs.
        
        Args:
            image: Input image
            tracks: List of track dictionaries
            color_map: Optional color mapping for track states
            
        Returns:
            Image with drawn tracking boxes
        """
        if color_map is None:
            color_map = {
                'confirmed': (0, 255, 0),    # Green
                'tentative': (0, 255, 255),   # Yellow
                'deleted': (0, 0, 255)        # Red
            }
        
        annotated = image.copy()
        
        for track in tracks:
            bbox = track.get('bbox', [])
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            track_id = track.get('track_id', 'N/A')
            state = track.get('state', 'tentative')
            confidence = track.get('confidence', 0.0)
            
            # Get color for this track state
            color = color_map.get(state, color_map['tentative'])
            
            # Create label with track ID
            label = f"ID: {track_id}"
            if confidence > 0:
                label += f" ({confidence:.2f})"
            
            # Draw tracking box
            annotated = self.draw_bounding_box(
                annotated, (x1, y1, x2, y2), label, confidence, color
            )
        
        return annotated
    
    def draw_face_boxes(
        self,
        image: np.ndarray,
        faces: List[Dict[str, Any]],
        show_landmarks: bool = False
    ) -> np.ndarray:
        """
        Draw face detection boxes with optional landmarks.
        
        Args:
            image: Input image
            faces: List of face detection dictionaries
            show_landmarks: Whether to draw facial landmarks
            
        Returns:
            Image with drawn face boxes
        """
        annotated = image.copy()
        
        for face in faces:
            bbox = face.get('bbox', [])
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            confidence = face.get('confidence', 0.0)
            person_name = face.get('person_name', 'Unknown')
            is_known = face.get('is_known', False)
            
            # Choose color based on recognition status
            color = (0, 255, 0) if is_known else (0, 0, 255)  # Green for known, Red for unknown
            
            # Create label
            label = person_name if is_known else "Unknown"
            if confidence > 0:
                label += f" ({confidence:.2f})"
            
            # Draw face box
            annotated = self.draw_bounding_box(
                annotated, (x1, y1, x2, y2), label, confidence, color
            )
            
            # Draw landmarks if available and requested
            if show_landmarks and 'landmarks' in face:
                landmarks = face['landmarks']
                if landmarks is not None and len(landmarks) > 0:
                    for point in landmarks:
                        cv2.circle(annotated, (int(point[0]), int(point[1])), 2, (255, 0, 255), -1)
        
        return annotated
    
    def resize_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize with aspect ratio
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create canvas with target size
            canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
            
            # Center the resized image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def preprocess_for_model(
        self,
        image: np.ndarray,
        input_size: Tuple[int, int] = (640, 640),
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image
            input_size: Model input size (width, height)
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image
        """
        # Resize image
        processed = self.resize_image(image, input_size, maintain_aspect=False)
        
        # Convert BGR to RGB
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Normalize if requested
        if normalize:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def save_image(
        self,
        image: np.ndarray,
        filepath: str,
        quality: int = 95
    ) -> bool:
        """
        Save image to file.
        
        Args:
            image: Image to save
            filepath: Output file path
            quality: JPEG quality (1-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Determine file format and save
            if filepath.lower().endswith('.jpg') or filepath.lower().endswith('.jpeg'):
                cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(filepath, image)
            
            logger.info(f"Image saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save image to {filepath}: {e}")
            return False
    
    def load_image(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load image from file.
        
        Args:
            filepath: Path to image file
            
        Returns:
            Loaded image or None if failed
        """
        try:
            image = cv2.imread(filepath)
            if image is None:
                logger.error(f"Could not load image: {filepath}")
                return None
            
            logger.info(f"Image loaded: {filepath}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {filepath}: {e}")
            return None
    
    def create_annotated_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]] = None,
        show_landmarks: bool = False
    ) -> np.ndarray:
        """
        Create fully annotated frame with detections and tracks.
        
        Args:
            frame: Input frame
            detections: List of detections
            tracks: Optional list of tracks
            show_landmarks: Whether to show facial landmarks
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw detections
        if detections:
            annotated = self.draw_multiple_boxes(annotated, detections)
        
        # Draw tracks
        if tracks:
            annotated = self.draw_tracking_boxes(annotated, tracks)
        
        # Draw face-specific annotations
        face_detections = [d for d in detections if d.get('type') == 'face']
        if face_detections:
            annotated = self.draw_face_boxes(annotated, face_detections, show_landmarks)
        
        return annotated
