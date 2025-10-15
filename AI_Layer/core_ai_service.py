"""
Core AI Service - Main orchestrator for AI Layer
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Ensure AI Layer directory is on sys.path for absolute subpackage imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger
from services.detection_service import DetectionService
from services.face_processing_service import FaceProcessingService
from services.vectorization_service import VectorizationService
from services.tracking_service import FaceTrackingService
from services.strongsort_tracking_service import StrongSORTTrackingService
from services.database_service import DatabaseService
from services.video_processing_service import VideoProcessingService
from services.insightface_recognition_service import InsightFaceRecognitionService
from utils.batch_face_processor import BatchFaceProcessor
from utils.image_processor import ImageProcessor
from utils.video_processor import VideoProcessor
from utils.file_manager import FileManager

logger = setup_logger(__name__)


class CoreAIService:
    """Main AI service orchestrator"""
    
    def __init__(self, config_path: str = None):
        """Initialize core AI service
        
        Accepts either a config path (str) or a ConfigLoader instance for backwards compatibility.
        """
        # Allow passing a ConfigLoader directly (backwards compatible with examples)
        if isinstance(config_path, ConfigLoader):
            self.config_loader = config_path
        else:
            self.config_loader = ConfigLoader(config_path)
        self.logger = logger
        
        # Initialize services
        self.detection_service = DetectionService(
            self.config_loader.get_detection_config(),
            enable_tracking=True
        )
        
        self.face_processing_service = FaceProcessingService(
            self.config_loader.get_face_processing_config()
        )
        
        # Share the InsightFace analyzer instance to avoid dual initialization
        shared_analyzer = getattr(self.face_processing_service, 'face_analyzer', None)
        self.vectorization_service = VectorizationService(
            self.config_loader.get_vectorization_config(),
            shared_face_analyzer=shared_analyzer
        )
        
        self.tracking_service = FaceTrackingService(
            self.config_loader.get_tracking_config()
        )
        
        # Initialize StrongSORT tracking if enabled
        strongsort_config = self.config_loader.get('strongsort_tracking', {})
        if strongsort_config.get('enabled', False):
            self.strongsort_service = StrongSORTTrackingService(strongsort_config)
        else:
            self.strongsort_service = None
        
        self.database_service = DatabaseService(
            self.config_loader.get_storage_config()
        )
        
        # Initialize InsightFace recognition service
        face_recognition_config = self.config_loader.get('face_recognition', {})
        if face_recognition_config.get('enabled', True):  # Enable by default
            self.face_recognition_service = InsightFaceRecognitionService(face_recognition_config)
        else:
            self.face_recognition_service = None
        
        # Pass video config augmented with data output path for stream CSV writing
        _video_cfg = dict(self.config_loader.get_video_config() or {})
        _data_out_cfg = self.config_loader.get_data_output_config() or {}
        if 'csv_path' in _data_out_cfg:
            _video_cfg['data_output_csv_path'] = _data_out_cfg.get('csv_path')
        self.video_processing_service = VideoProcessingService(_video_cfg)
        
        self.batch_processor = BatchFaceProcessor(self.config_loader)
        
        # Initialize utility services
        self.image_processor = ImageProcessor()
        self.video_processor = VideoProcessor()
        self.file_manager = FileManager(self.config_loader.get_paths_config())
        
        self.logger.info("Core AI Service initialized successfully")
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Process single image for face detection and recognition"""
        try:
            # Detect faces
            face_detections = self.detection_service.detect_faces(image)
            
            if not face_detections:
                return {
                    'success': True,
                    'faces_detected': 0,
                    'faces_processed': 0,
                    'recognitions': []
                }
            
            # Process faces
            processed_faces = self.face_processing_service.process_face_detections(
                image, face_detections
            )
            
            # Apply tracking
            tracked_faces = self.tracking_service.update(face_detections)
            
            # Extract embeddings and recognize
            recognitions = []
            for i, face_info in enumerate(processed_faces):
                face_crop = face_info['face_crop']
                # Use improved vectorization for pre-cropped faces
                embedding = self.vectorization_service.extract_embedding_from_crop(face_crop)
                
                if embedding is not None:
                    # For now, return basic recognition info
                    # In full implementation, would compare against reference embeddings
                    recognition = {
                        'face_id': i,
                        'bbox': face_info['original_bbox'],
                        'confidence': face_info['confidence'],
                        'embedding_available': True,
                        'track_id': tracked_faces[i].get('track_id', 'N/A') if i < len(tracked_faces) else 'N/A'
                    }
                    recognitions.append(recognition)
            
            return {
                'success': True,
                'faces_detected': len(face_detections),
                'faces_processed': len(processed_faces),
                'recognitions': recognitions
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_video_frame(self, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """Process single video frame"""
        result = self.process_image(frame)
        result['frame_id'] = frame_id
        return result
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all AI services"""
        return {
            'detection_service': {
                'available': self.detection_service.is_model_loaded(),
                'models_loaded': {
                    'face_model': self.detection_service.face_model is not None,
                    'human_model': self.detection_service.human_model is not None
                }
            },
            'face_processing_service': {
                'available': True,
                'alignment_available': self.face_processing_service.is_alignment_available()
            },
            'vectorization_service': {
                'available': self.vectorization_service.is_available()
            },
            'tracking_service': {
                'available': True,
                'active_tracks': len(self.tracking_service.tracks),
                'statistics': self.tracking_service.get_tracking_statistics()
            }
        }
    
    def reset_tracking(self):
        """Reset tracking service"""
        self.tracking_service.reset()
        self.logger.info("Tracking service reset")
    
    def process_video_with_face_detection(self, video_path: str) -> Dict[str, Any]:
        """Process video for face detection and recognition"""
        try:
            # Load reference embeddings from database
            reference_embeddings = self.database_service.get_all_embeddings()
            
            # Process video
            result = self.video_processing_service.process_video_with_face_detection(
                video_path=video_path,
                detection_service=self.detection_service,
                face_processing_service=self.face_processing_service,
                vectorization_service=self.vectorization_service,
                tracking_service=self.tracking_service,
                database_service=self.database_service,
                reference_embeddings=reference_embeddings
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing video with face detection: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_video_with_human_detection(self, video_path: str) -> Dict[str, Any]:
        """Process video for human detection and ReID tracking"""
        try:
            # Process video
            result = self.video_processing_service.process_video_with_human_detection(
                video_path=video_path,
                detection_service=self.detection_service,
                tracking_service=self.tracking_service,
                database_service=self.database_service
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing video with human detection: {e}")
            return {'success': False, 'error': str(e)}
    
    def store_face_vector(
        self,
        person_id: str,
        person_name: str,
        embedding: np.ndarray,
        bbox: List[float],
        confidence: float,
        image_path: str = None,
        face_crop_path: str = None,
        status: str = None,
        location: str = None
    ) -> bool:
        """Store face vector in database"""
        return self.database_service.store_face_vector(
            person_id=person_id,
            person_name=person_name,
            embedding=embedding,
            bbox=bbox,
            confidence=confidence,
            image_path=image_path,
            face_crop_path=face_crop_path,
            status=status,
            location=location
        )
    
    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all face embeddings from database"""
        return self.database_service.get_all_embeddings()
    
    def update_person_status(self, person_id: str, status: str = None, location: str = None) -> bool:
        """Update person status and location"""
        return self.database_service.update_person_status(person_id, status, location)
    
    def get_persons_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all persons with specific status"""
        return self.database_service.get_persons_by_status(status)
    
    def get_persons_by_location(self, location: str) -> List[Dict[str, Any]]:
        """Get all persons from specific location"""
        return self.database_service.get_persons_by_location(location)
    
    def get_person_by_id(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get person information by ID"""
        return self.database_service.get_person_by_id(person_id)
    
    def process_face_folder(self, folder_path: str = None) -> Dict[str, Any]:
        """Process a folder of face images for batch vectorization"""
        return self.batch_processor.process_folder(folder_path)
    
    def get_batch_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about batch processed faces"""
        return self.batch_processor.get_processing_stats()
    
    def cleanup_duplicate_faces(self, similarity_threshold: float = None) -> Dict[str, Any]:
        """Remove duplicate faces from database"""
        return self.batch_processor.cleanup_duplicates(similarity_threshold)
    
    def draw_bounding_boxes(
        self, 
        image: np.ndarray, 
        detections: List[Dict[str, Any]],
        show_labels: bool = True
    ) -> np.ndarray:
        """Draw bounding boxes on image"""
        return self.image_processor.draw_multiple_boxes(image, detections)
    
    def draw_face_boxes(
        self, 
        image: np.ndarray, 
        faces: List[Dict[str, Any]],
        show_landmarks: bool = False
    ) -> np.ndarray:
        """Draw face detection boxes with optional landmarks"""
        return self.image_processor.draw_face_boxes(image, faces, show_landmarks)
    
    def draw_tracking_boxes(
        self, 
        image: np.ndarray, 
        tracks: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Draw tracking boxes with track IDs"""
        return self.image_processor.draw_tracking_boxes(image, tracks)
    
    def create_annotated_frame(
        self, 
        frame: np.ndarray, 
        detections: List[Dict[str, Any]], 
        tracks: List[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Create fully annotated frame with detections and tracks"""
        return self.image_processor.create_annotated_frame(frame, detections, tracks)
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information"""
        return self.video_processor.get_video_info(video_path)
    
    def extract_frames(
        self, 
        video_path: str, 
        output_dir: str, 
        frame_skip: int = 1
    ) -> List[str]:
        """Extract frames from video"""
        return self.video_processor.extract_frames(video_path, output_dir, frame_skip)
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get file storage statistics"""
        return self.file_manager.get_storage_statistics()
    
    def cleanup_old_files(self, days_old: int = 30, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up old files"""
        return self.file_manager.cleanup_old_files(days_old, dry_run)
    
    def backup_data(self, backup_path: str) -> bool:
        """Create backup of all data"""
        return self.file_manager.backup_data(backup_path)
    
    def update_strongsort_tracking(
        self, 
        detections: List[Dict[str, Any]], 
        frame: np.ndarray,
        embeddings: List[Optional[np.ndarray]] = None
    ) -> List[Dict[str, Any]]:
        """Update StrongSORT tracking with detections"""
        if self.strongsort_service is None:
            return []
        return self.strongsort_service.update(detections, frame, embeddings)
    
    def get_strongsort_statistics(self) -> Dict[str, Any]:
        """Get StrongSORT tracking statistics"""
        if self.strongsort_service is None:
            return {'error': 'StrongSORT not enabled'}
        return self.strongsort_service.get_tracking_statistics()
    
    def get_active_strongsort_tracks(self) -> List[Dict[str, Any]]:
        """Get active StrongSORT tracks"""
        if self.strongsort_service is None:
            return []
        return self.strongsort_service.get_active_tracks()
    
    def get_strongsort_track_by_id(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get StrongSORT track by ID"""
        if self.strongsort_service is None:
            return None
        return self.strongsort_service.get_track_by_id(track_id)
    
    def reset_strongsort_tracker(self) -> bool:
        """Reset StrongSORT tracker"""
        if self.strongsort_service is None:
            return False
        self.strongsort_service.reset_tracker()
        return True
    
    def is_strongsort_available(self) -> bool:
        """Check if StrongSORT is available"""
        if self.strongsort_service is None:
            return False
        return self.strongsort_service.is_available()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            'detection': self.config_loader.get_detection_config(),
            'face_processing': self.config_loader.get_face_processing_config(),
            'vectorization': self.config_loader.get_vectorization_config(),
            'tracking': self.config_loader.get_tracking_config(),
            'storage': self.config_loader.get_storage_config(),
            'paths': self.config_loader.get_paths_config(),
            'video': self.config_loader.get_video_config()
        }
