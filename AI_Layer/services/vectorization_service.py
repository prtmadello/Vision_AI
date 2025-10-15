"""
Face vectorization service for AI Layer
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
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


class VectorizationService:
    """Face vectorization service using ArcFace"""
    
    def __init__(self, config: Dict[str, Any], shared_face_analyzer: Any = None):
        self.config = config
        self.logger = logger
        self.face_analyzer = shared_face_analyzer
        
        if self.face_analyzer is None:
            if INSIGHTFACE_AVAILABLE:
                self._initialize_insightface()
            else:
                self.logger.error("InsightFace not available. Cannot initialize vectorization service.")
    
    def _initialize_insightface(self) -> None:
        """Initialize InsightFace for face vectorization"""
        try:
            model_name = self.config.get('model_name', 'buffalo_l')
            self.face_analyzer = FaceAnalysis(name=model_name)
            # Lower detection threshold for better face detection
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
            self.logger.info(f"InsightFace initialized with model: {model_name} (det_thresh=0.3)")
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            self.face_analyzer = None
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from properly cropped and aligned face image using ArcFace"""
        if self.face_analyzer is None:
            self.logger.error("Face analyzer not initialized")
            return None
        
        try:
            # Use InsightFace's built-in face analysis for proper face detection and alignment
            faces = self.face_analyzer.get(face_image)
            
            if not faces:
                self.logger.warning(f"No faces detected in image for vectorization (image shape: {face_image.shape})")
                return None
            
            self.logger.debug(f"InsightFace detected {len(faces)} faces in image")
            
            # Use the first (largest) face
            face = faces[0]
            
            # Get the embedding directly from InsightFace (already processed and aligned)
            # InsightFace handles face detection, alignment, and feature extraction internally
            embedding = face.embedding
            
            # Normalize if configured
            if self.config.get('normalize_vectors', True):
                embedding = embedding / np.linalg.norm(embedding)
            
            self.logger.debug(f"Extracted embedding with dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def extract_embedding_from_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding from a pre-cropped face image (for use with YOLO detections)"""
        if self.face_analyzer is None:
            self.logger.error("Face analyzer not initialized")
            return None
        
        try:
            # For pre-cropped faces, we need to ensure proper preprocessing
            # InsightFace expects the full image, but we can work with crops too
            
            # Ensure the crop is large enough for good feature extraction
            h, w = face_crop.shape[:2]
            min_size = self.config.get('min_face_size', 32)
            if h < min_size or w < min_size:
                self.logger.warning(f"Face crop too small for vectorization: {w}x{h} (min: {min_size})")
                return None
            
            # Use InsightFace on the cropped face
            faces = self.face_analyzer.get(face_crop)
            
            if not faces:
                self.logger.warning("No faces detected in cropped image for vectorization")
                return None
            
            # Use the first face
            face = faces[0]
            embedding = face.embedding
            
            # Normalize if configured
            if self.config.get('normalize_vectors', True):
                embedding = embedding / np.linalg.norm(embedding)
            
            self.logger.debug(f"Extracted embedding from crop with dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting embedding from crop: {e}")
            return None
    
    def compare_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compare two face embeddings using cosine similarity"""
        try:
            if self.config.get('normalize_vectors', True):
                emb1 = embedding1 / np.linalg.norm(embedding1)
                emb2 = embedding2 / np.linalg.norm(embedding2)
            else:
                emb1 = embedding1
                emb2 = embedding2
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error comparing embeddings: {e}")
            return 0.0
    
    def find_best_match(
        self,
        query_embedding: np.ndarray,
        reference_embeddings: List[Dict[str, Any]],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Find best matching face from reference embeddings"""
        if threshold is None:
            threshold = self.config.get('matching_threshold', 0.6)
        
        best_match = {
            'person_id': None,
            'confidence': 0.0,
            'is_known': False
        }
        best_ref_data = None
        
        for ref_data in reference_embeddings:
            ref_embedding = ref_data['embedding']
            similarity = self.compare_embeddings(query_embedding, ref_embedding)
            
            if similarity > best_match['confidence']:
                best_match['confidence'] = similarity
                best_match['person_id'] = ref_data['person_id']
                best_ref_data = ref_data
        
        if best_match['confidence'] >= threshold:
            best_match['is_known'] = True
            if best_ref_data:
                person_name = best_ref_data.get('person_name', 'Unknown')
                if not person_name or person_name == 'Unknown':
                    person_id = best_match['person_id']
                    if '_' in person_id:
                        parts = person_id.split('_')
                        name_parts = []
                        for part in parts:
                            if len(part) == 8 and part.isdigit():
                                break
                            name_parts.append(part)
                        person_name = '_'.join(name_parts) if name_parts else person_id
                    else:
                        person_name = person_id
                best_match['person_name'] = person_name
                if 'status' in best_ref_data:
                    best_match['status'] = best_ref_data.get('status')
        
        return best_match
    
    def process_face_for_recognition(
        self,
        face_image: np.ndarray,
        reference_embeddings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process face for recognition against reference embeddings"""
        embedding = self.extract_embedding(face_image)
        
        if embedding is None:
            return {
                'is_known': False,
                'person_id': None,
                'person_name': 'Unknown',
                'confidence': 0.0
            }
        
        match_result = self.find_best_match(embedding, reference_embeddings)
        return match_result
    
    def is_available(self) -> bool:
        """Check if vectorization service is available"""
        return self.face_analyzer is not None
