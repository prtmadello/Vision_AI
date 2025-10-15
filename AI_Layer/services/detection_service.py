"""
Enhanced Face and Human Detection Service for AI Layer
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available. Install ultralytics package.")

from utils.logger import setup_logger

logger = setup_logger(__name__)


class DetectionService:
    """Enhanced Face and Human Detection Service"""
    
    def __init__(self, config: Dict[str, Any], enable_tracking: bool = True):
        self.config = config
        self.enable_tracking = enable_tracking
        self.logger = logger
        
        self.face_model = None
        self.human_model = None
        
        if YOLO_AVAILABLE:
            self._load_models()
        else:
            self.logger.error("YOLO not available. Cannot initialize detection service.")
    
    def _load_models(self) -> None:
        """Load YOLO models based on detection mode"""
        try:
            detection_mode = self.config.get('detection_mode', 'both')
            
            if detection_mode in ["face_only", "both"]:
                face_model_path = self.config.get('face_model_path')
                if Path(face_model_path).exists():
                    self.face_model = YOLO(face_model_path)
                    self.logger.info(f"Face model loaded: {face_model_path}")
                else:
                    self.logger.error(f"Face model not found: {face_model_path}")
            
            if detection_mode in ["human_only", "both"]:
                human_model_path = self.config.get('human_model_path')
                if Path(human_model_path).exists():
                    self.human_model = YOLO(human_model_path)
                    self.logger.info(f"Human model loaded: {human_model_path}")
                else:
                    self.logger.error(f"Human model not found: {human_model_path}")
                    
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image"""
        detections = self._detect_objects_single(image)
        face_detections = [d for d in detections if d.get('detection_type') == 'face']
        
        self.logger.info(f"Raw face detections: {len(face_detections)}")
        
        validated_detections = []
        for detection in face_detections:
            if self._validate_face_detection(detection, image.shape):
                validated_detections.append(detection)
            else:
                self.logger.info(f"Face detection filtered out: bbox={detection.get('bbox')}, confidence={detection.get('confidence')}")
        
        self.logger.info(f"Validated face detections: {len(validated_detections)}")
        return validated_detections
    
    def _detect_objects_single(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces and/or humans in a single image"""
        all_detections = []
        detection_mode = self.config.get('detection_mode', 'both')
        
        if detection_mode in ["face_only", "both"] and self.face_model is not None:
            face_detections = self._detect_faces_yolo(image)
            all_detections.extend(face_detections)
        
        if detection_mode in ["human_only", "both"] and self.human_model is not None:
            human_detections = self._detect_humans_yolo(image)
            all_detections.extend(human_detections)
        
        return all_detections
    
    def _detect_faces_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using YOLO face model"""
        try:
            results = self.face_model(
                image,
                conf=self.config.get('face_confidence_threshold', 0.5),
                iou=self.config.get('face_iou_threshold', 0.45),
                max_det=self.config.get('face_max_detections', 100)
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if class_id == 0:
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': 'face',
                                'detection_type': 'face',
                                'detection_method': 'yolo_face'
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in face detection: {e}")
            return []
    
    def _detect_humans_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect humans using YOLO general model"""
        try:
            results = self.human_model(
                image,
                conf=self.config.get('human_confidence_threshold', 0.5),
                iou=self.config.get('human_iou_threshold', 0.45),
                max_det=self.config.get('human_max_detections', 100)
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if class_id == 0:
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': 'person',
                                'detection_type': 'human',
                                'detection_method': 'yolo_human'
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in human detection: {e}")
            return []
    
    def _validate_face_detection(self, detection: Dict[str, Any], image_shape: Tuple[int, int]) -> bool:
        """Validate face detection to filter out false positives"""
        bbox = detection.get('bbox', [0, 0, 0, 0])
        confidence = detection.get('confidence', 0.0)
        
        if len(bbox) != 4:
            return False
        
        x1, y1, x2, y2 = bbox
        
        if x1 >= x2 or y1 >= y2:
            return False
        
        height, width = image_shape[:2]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False
        
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if bbox_width < 10 or bbox_height < 10:
            return False
        
        aspect_ratio = bbox_width / bbox_height
        if aspect_ratio < 0.6 or aspect_ratio > 1.7:
            return False
        
        if confidence < self.config.get('face_confidence_threshold', 0.5):
            return False
        
        return True
    
    def is_model_loaded(self) -> bool:
        """Check if any model is loaded"""
        detection_mode = self.config.get('detection_mode', 'both')
        if detection_mode == "face_only":
            return self.face_model is not None
        elif detection_mode == "human_only":
            return self.human_model is not None
        else:
            return self.face_model is not None or self.human_model is not None
