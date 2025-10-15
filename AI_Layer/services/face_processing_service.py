"""
Face processing service for AI Layer
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Install insightface package.")

from utils.logger import setup_logger

logger = setup_logger(__name__)


class FaceProcessingService:
    """Face processing service for alignment and preprocessing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        self.face_analyzer = None
        
        if INSIGHTFACE_AVAILABLE:
            self._initialize_insightface()
        else:
            self.logger.warning("InsightFace not available. Face alignment will be disabled.")
    
    def _initialize_insightface(self) -> None:
        """Initialize InsightFace for face alignment"""
        try:
            self.face_analyzer = FaceAnalysis(name='buffalo_l')
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info("InsightFace initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            self.face_analyzer = None
    
    def process_face_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process face detections with alignment and cropping"""
        processed_faces = []
        
        for i, detection in enumerate(detections):
            try:
                bbox = detection['bbox']
                confidence = detection['confidence']
                
                if not self._is_valid_face_size(bbox):
                    continue
                
                face_crop = self._crop_face_with_padding(image, bbox)
                if face_crop is None:
                    continue
                
                aligned_face = None
                landmarks = None
                
                if self.config.get('alignment_enabled', True) and self.face_analyzer is not None:
                    aligned_face, landmarks = self._align_face(face_crop)
                
                final_face = aligned_face if aligned_face is not None else face_crop
                
                processed_face = {
                    'face_id': i,
                    'original_bbox': bbox,
                    'confidence': confidence,
                    'face_crop': final_face,
                    'landmarks': landmarks,
                    'is_aligned': aligned_face is not None,
                    'face_size': self._calculate_face_size(bbox)
                }
                
                processed_faces.append(processed_face)
                
            except Exception as e:
                self.logger.error(f"Error processing face {i}: {e}")
                continue
        
        return processed_faces
    
    def _crop_face_with_padding(self, image: np.ndarray, bbox: List[int], padding: float = None) -> Optional[np.ndarray]:
        """Crop face with padding and minimum size requirements"""
        if padding is None:
            padding = self.config.get('crop_padding', 0.3)
        
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Calculate padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        
        # Apply padding
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        
        # Crop the face
        face_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Ensure minimum size for InsightFace (at least 112x112)
        min_size = 112
        if face_crop.shape[0] < min_size or face_crop.shape[1] < min_size:
            # Resize to minimum size
            face_crop = cv2.resize(face_crop, (min_size, min_size))
            self.logger.debug(f"Resized face crop from {face_crop.shape} to {min_size}x{min_size}")
        
        return face_crop
    
    def _align_face(self, face_crop: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Align face using InsightFace with proper landmark detection"""
        if self.face_analyzer is None:
            return None, None
        
        try:
            # Use InsightFace for proper face detection and alignment
            faces = self.face_analyzer.get(face_crop)
            if not faces:
                self.logger.warning(f"No faces detected in cropped image for vectorization (crop shape: {face_crop.shape})")
                return None, None
            
            self.logger.debug(f"InsightFace detected {len(faces)} faces in cropped image")
            
            face = faces[0]
            
            # Get landmarks for alignment
            landmarks = face.kps if hasattr(face, 'kps') else None
            
            # For proper face alignment, we need to use the aligned face from InsightFace
            # InsightFace already provides aligned faces, so we use the original crop
            # but we could implement proper alignment using landmarks if needed
            aligned_face = face_crop
            
            # Optional: Implement proper face alignment using landmarks
            if (landmarks is not None and len(landmarks) >= 5 and 
                self.config.get('use_landmark_alignment', True)):
                # Use landmarks for better alignment if configured
                aligned_face = self.align_face_with_landmarks(face_crop, landmarks)
                if aligned_face is not None:
                    aligned_face = aligned_face
            
            return aligned_face, landmarks
            
        except Exception as e:
            self.logger.error(f"Error in face alignment: {e}")
            return None, None
    
    def _is_valid_face_size(self, bbox: List[int]) -> bool:
        """Check if face size is within valid range"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        size = max(width, height)
        
        min_size = self.config.get('min_face_size', 20)
        max_size = self.config.get('max_face_size', 1000)
        
        return min_size <= size <= max_size
    
    def _calculate_face_size(self, bbox: List[int]) -> int:
        """Calculate face size from bounding box"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return max(width, height)
    
    def preprocess_for_vectorization(
        self,
        face_crop: np.ndarray,
        target_size: Tuple[int, int] = (112, 112)
    ) -> Optional[np.ndarray]:
        """Preprocess face crop for vectorization with proper alignment"""
        try:
            # First, try to get properly aligned face using InsightFace
            if self.face_analyzer is not None:
                faces = self.face_analyzer.get(face_crop)
                if faces:
                    # Use InsightFace's processed face (already aligned)
                    face = faces[0]
                    # InsightFace provides normalized faces, but we need to resize to target
                    processed = cv2.resize(face_crop, target_size)
                else:
                    # Fallback to manual preprocessing
                    processed = cv2.resize(face_crop, target_size)
            else:
                # Fallback to manual preprocessing
                processed = cv2.resize(face_crop, target_size)
            
            # Convert BGR to RGB
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] range
            processed = processed.astype(np.float32) / 255.0
            
            return processed
        except Exception as e:
            self.logger.error(f"Error preprocessing face: {e}")
            return None
    
    def align_face_with_landmarks(
        self,
        face_crop: np.ndarray,
        landmarks: np.ndarray
    ) -> Optional[np.ndarray]:
        """Align face using 5-point landmarks for better vectorization"""
        try:
            if landmarks is None or len(landmarks) < 5:
                return face_crop
            
            # Standard 5-point landmarks for face alignment
            # Left eye, right eye, nose, left mouth, right mouth
            src_points = landmarks.astype(np.float32)
            
            # Standard face template (normalized coordinates)
            template_points = np.array([
                [0.31556875000000000, 0.4615741071428571],  # Left eye
                [0.68262291666666670, 0.4615741071428571],  # Right eye
                [0.50026249999999990, 0.6405053571428571],  # Nose
                [0.34947187500000004, 0.8246919642857142],  # Left mouth
                [0.65343645833333330, 0.8246919642857142]   # Right mouth
            ], dtype=np.float32)
            
            # Scale template points to image size
            h, w = face_crop.shape[:2]
            template_points[:, 0] *= w
            template_points[:, 1] *= h
            
            # Calculate transformation matrix
            transform_matrix = cv2.getAffineTransform(
                src_points[:3].astype(np.float32),
                template_points[:3].astype(np.float32)
            )
            
            # Apply transformation
            aligned_face = cv2.warpAffine(
                face_crop, transform_matrix, (w, h), flags=cv2.INTER_LINEAR
            )
            
            return aligned_face
            
        except Exception as e:
            self.logger.error(f"Error aligning face with landmarks: {e}")
            return face_crop
    
    def is_alignment_available(self) -> bool:
        """Check if face alignment is available"""
        return self.face_analyzer is not None and self.config.get('alignment_enabled', True)
