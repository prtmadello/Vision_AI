"""
InsightFace Recognition Service for Vision AI system.
Lightweight face recognition service that integrates with existing tracking.
Only runs recognition when good face features are available.
"""

import cv2
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import time

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Install insightface package.")

from utils.logger import setup_logger

logger = setup_logger(__name__)


class InsightFaceRecognitionService:
    """InsightFace recognition service with performance filters."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize InsightFace recognition service.
        
        Args:
            config: Recognition configuration
        """
        self.config = config
        self.logger = logger
        
        if not INSIGHTFACE_AVAILABLE:
            self.logger.error("InsightFace not available. Install insightface package.")
            self.app = None
            return
        
        # Recognition parameters
        self.recognition_threshold = config.get('recognition_threshold', 0.15)
        self.embeddings_path = config.get('embeddings_path', 'face_embeddings.json')
        self.min_face_size = config.get('min_face_size', 50)  # Minimum face size for recognition
        self.max_faces_per_frame = config.get('max_faces_per_frame', 3)  # Performance limit
        self.recognition_interval = config.get('recognition_interval', 5)  # Run every N frames
        
        # Performance tracking
        self.frame_count = 0
        self.last_recognition_time = 0
        self.recognition_cache = {}  # Cache recent recognitions
        
        # Initialize InsightFace
        self._initialize_insightface()
        
        # Load reference embeddings
        self.reference_embeddings = self._load_reference_embeddings()
        
        self.logger.info("InsightFace recognition service initialized")
    
    def _initialize_insightface(self) -> None:
        """Initialize InsightFace application."""
        try:
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info("InsightFace initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            self.app = None
    
    def _load_reference_embeddings(self) -> List[Dict[str, Any]]:
        """Load reference embeddings from JSON file."""
        if not os.path.exists(self.embeddings_path):
            self.logger.warning(f"Embeddings file not found: {self.embeddings_path}")
            return []
        
        try:
            with open(self.embeddings_path, 'r') as f:
                data = json.load(f)
            
            embeddings = data.get('embeddings', [])
            self.logger.info(f"Loaded {len(embeddings)} reference embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            return []
    
    def _should_run_recognition(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> bool:
        """
        Determine if recognition should run based on performance filters.
        
        Args:
            frame: Current frame
            detections: Human detections from YOLO
            
        Returns:
            bool: True if recognition should run
        """
        # Check frame interval
        if self.frame_count % self.recognition_interval != 0:
            return False
        
        # Check if we have reasonable number of detections
        if len(detections) > self.max_faces_per_frame:
            return False
        
        # Check if detections have reasonable size (likely to contain faces)
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                if width >= self.min_face_size and height >= self.min_face_size:
                    return True
        
        return False
    
    def _extract_face_embedding(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract face embedding from a bounding box region.
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Optional[np.ndarray]: Face embedding or None
        """
        if self.app is None:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            
            # Add some padding to the bounding box
            padding = 0.2
            h, w = frame.shape[:2]
            pad_w = int((x2 - x1) * padding)
            pad_h = int((y2 - y1) * padding)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            # Extract face region
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return None
            
            # Detect faces in the region
            faces = self.app.get(face_region)
            
            if len(faces) == 0:
                return None
            
            # Use the largest face
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            return largest_face.embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def _compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings using cosine similarity."""
        try:
            # Normalize embeddings
            emb1 = embedding1 / np.linalg.norm(embedding1)
            emb2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2)
            similarity = max(0.0, min(1.0, similarity))
            
            # Add small tolerance boost for better recognition
            if similarity > 0.1:
                similarity = min(1.0, similarity + 0.05)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error comparing embeddings: {e}")
            return 0.0
    
    def _recognize_face(self, embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Recognize face against reference embeddings with enhanced validation.
        
        Args:
            embedding: Face embedding to recognize
            
        Returns:
            Optional[Dict]: Recognition result or None
        """
        if not self.reference_embeddings:
            return None
        
        best_match = {
            'person_name': 'Unknown',
            'confidence': 0.0,
            'status': 'normal',
            'person_label': 'customer'
        }
        
        second_best_match = {
            'person_name': 'Unknown',
            'confidence': 0.0,
            'status': 'normal',
            'person_label': 'customer'
        }
        
        # Find best and second best matches
        for ref_data in self.reference_embeddings:
            ref_embedding = np.array(ref_data['embedding'])
            similarity = self._compare_embeddings(embedding, ref_embedding)
            
            if similarity > best_match['confidence']:
                # Move current best to second best
                second_best_match = best_match.copy()
                
                # Update best match
                best_match['confidence'] = similarity
                best_match['person_name'] = ref_data['person_name']
                best_match['status'] = ref_data.get('status', 'normal')
                best_match['person_label'] = ref_data.get('person_label', 'customer')
            elif similarity > second_best_match['confidence']:
                # Update second best match
                second_best_match['confidence'] = similarity
                second_best_match['person_name'] = ref_data['person_name']
                second_best_match['status'] = ref_data.get('status', 'normal')
                second_best_match['person_label'] = ref_data.get('person_label', 'customer')
        
        # Clean up person name (remove "_face_0" suffix)
        if best_match['person_name'] != 'Unknown' and '_face_' in best_match['person_name']:
            best_match['person_name'] = best_match['person_name'].split('_face_')[0]
        
        # Enhanced validation: Check if there's a clear winner
        confidence_gap = best_match['confidence'] - second_best_match['confidence']
        min_confidence_gap = 0.1  # Minimum gap to ensure clear distinction
        
        # Only return if:
        # 1. Confidence is above threshold
        # 2. There's a clear gap between best and second best
        # 3. The confidence is not too close to the threshold (add safety margin)
        safety_margin = 0.05
        if (best_match['confidence'] >= (self.recognition_threshold + safety_margin) and 
            confidence_gap >= min_confidence_gap):
            
            self.logger.debug(f"Face recognized: {best_match['person_name']} "
                            f"(confidence: {best_match['confidence']:.3f}, "
                            f"gap: {confidence_gap:.3f})")
            return best_match
        else:
            if best_match['confidence'] >= self.recognition_threshold:
                self.logger.debug(f"Face recognition uncertain: {best_match['person_name']} "
                                f"(confidence: {best_match['confidence']:.3f}, "
                                f"gap: {confidence_gap:.3f} < {min_confidence_gap})")
            return None
    
    def recognize_faces_in_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Recognize faces in human detections with performance filters.
        
        Args:
            frame: Current frame
            detections: Human detections from YOLO
            
        Returns:
            Dict[int, Dict]: Mapping of detection index to recognition data
        """
        self.frame_count += 1
        
        # Check if we should run recognition
        if not self._should_run_recognition(frame, detections):
            return {}
        
        recognitions = {}
        
        try:
            for i, detection in enumerate(detections):
                bbox = detection.get('bbox', [])
                if len(bbox) < 4:
                    continue
                
                # Extract face embedding
                embedding = self._extract_face_embedding(frame, bbox)
                if embedding is None:
                    continue
                
                # Recognize face
                recognition = self._recognize_face(embedding)
                if recognition:
                    recognitions[i] = recognition
                    status_msg = f" [BLOCKED]" if recognition['status'] == 'blocked' else ""
                    self.logger.info(f"Recognized: {recognition['person_name']}{status_msg} (confidence: {recognition['confidence']:.3f})")
        
        except Exception as e:
            self.logger.error(f"Error in face recognition: {e}")
        
        return recognitions
    
    def get_recognition_stats(self) -> Dict[str, Any]:
        """Get recognition performance statistics."""
        return {
            'frame_count': self.frame_count,
            'reference_embeddings_count': len(self.reference_embeddings),
            'recognition_threshold': self.recognition_threshold,
            'cache_size': len(self.recognition_cache)
        }
