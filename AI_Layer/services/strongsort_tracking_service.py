"""
StrongSORT tracking service for Vision AI system.
Enhanced tracking with OSNet ReID for robust person tracking.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import time
from datetime import datetime

try:
    from boxmot import StrongSORT
    STRONGSORT_AVAILABLE = True
except ImportError:
    STRONGSORT_AVAILABLE = False
    logging.warning("StrongSORT not available. Install boxmot package.")

from utils.logger import setup_logger

logger = setup_logger(__name__)


class TrackReferenceSystem:
    """Enhanced track reference system to prevent ID switches when people come close."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        
        # Track reference parameters
        self.max_history_frames = config.get('max_history_frames', 20)
        self.spatial_consistency_threshold = config.get('spatial_consistency_threshold', 0.8)
        self.motion_consistency_threshold = config.get('motion_consistency_threshold', 0.7)
        self.embedding_consistency_threshold = config.get('embedding_consistency_threshold', 0.6)
        self.min_track_age_for_stability = config.get('min_track_age_for_stability', 10)
        
        # Track reference storage
        self.track_references = {}  # track_id -> TrackReference
        self.frame_references = []  # List of frame references for temporal analysis
        
    def update_track_reference(self, track_id: int, bbox: List[float], embedding: Optional[np.ndarray] = None, 
                             confidence: float = 1.0) -> None:
        """Update track reference with new detection data."""
        if track_id not in self.track_references:
            self.track_references[track_id] = TrackReference(track_id, self.max_history_frames)
        
        self.track_references[track_id].add_detection(bbox, embedding, confidence)
    
    def validate_track_assignment(self, track_id: int, detection_bbox: List[float], 
                                detection_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Validate if a track assignment is consistent with historical data."""
        if track_id not in self.track_references:
            return {'valid': True, 'confidence': 1.0, 'reason': 'new_track'}
        
        reference = self.track_references[track_id]
        
        # Check if track is old enough for stability validation
        if reference.get_age() < self.min_track_age_for_stability:
            return {'valid': True, 'confidence': 0.8, 'reason': 'young_track'}
        
        # Calculate consistency scores
        spatial_score = reference.calculate_spatial_consistency(detection_bbox)
        motion_score = reference.calculate_motion_consistency(detection_bbox)
        embedding_score = 1.0
        
        if detection_embedding is not None and reference.has_embeddings():
            embedding_score = reference.calculate_embedding_consistency(detection_embedding)
        
        # Overall consistency score
        overall_score = (
            spatial_score * 0.4 + 
            motion_score * 0.4 + 
            embedding_score * 0.2
        )
        
        is_valid = overall_score >= self.spatial_consistency_threshold
        
        return {
            'valid': is_valid,
            'confidence': overall_score,
            'spatial_score': spatial_score,
            'motion_score': motion_score,
            'embedding_score': embedding_score,
            'reason': 'validated' if is_valid else 'inconsistent'
        }
    
    def get_track_prediction(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get predicted position for a track based on motion history."""
        if track_id not in self.track_references:
            return None
        
        return self.track_references[track_id].predict_next_position()
    
    def cleanup_old_references(self, active_track_ids: set) -> None:
        """Clean up references for tracks that are no longer active."""
        tracks_to_remove = []
        for track_id in self.track_references:
            if track_id not in active_track_ids:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_references[track_id]


class TrackReference:
    """Individual track reference with historical data."""
    
    def __init__(self, track_id: int, max_history: int = 20):
        self.track_id = track_id
        self.max_history = max_history
        self.bbox_history = []
        self.embedding_history = []
        self.confidence_history = []
        self.timestamps = []
        self.creation_time = time.time()
    
    def add_detection(self, bbox: List[float], embedding: Optional[np.ndarray] = None, 
                     confidence: float = 1.0) -> None:
        """Add new detection to track history."""
        self.bbox_history.append(bbox.copy())
        self.confidence_history.append(confidence)
        self.timestamps.append(time.time())
        
        if embedding is not None:
            self.embedding_history.append(embedding.copy())
        
        # Maintain history size
        if len(self.bbox_history) > self.max_history:
            self.bbox_history.pop(0)
            self.confidence_history.pop(0)
            self.timestamps.pop(0)
            if self.embedding_history:
                self.embedding_history.pop(0)
    
    def get_age(self) -> int:
        """Get track age in frames."""
        return len(self.bbox_history)
    
    def has_embeddings(self) -> bool:
        """Check if track has embedding history."""
        return len(self.embedding_history) > 0
    
    def calculate_spatial_consistency(self, new_bbox: List[float]) -> float:
        """Calculate spatial consistency score based on position history."""
        if len(self.bbox_history) < 2:
            return 1.0
        
        # Calculate expected position based on recent movement
        recent_bboxes = self.bbox_history[-3:]  # Last 3 positions
        if len(recent_bboxes) < 2:
            return 1.0
        
        # Calculate average velocity
        velocities = []
        for i in range(1, len(recent_bboxes)):
            prev_center = self._get_bbox_center(recent_bboxes[i-1])
            curr_center = self._get_bbox_center(recent_bboxes[i])
            velocities.append([curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]])
        
        if not velocities:
            return 1.0
        
        avg_velocity = np.mean(velocities, axis=0)
        last_center = self._get_bbox_center(recent_bboxes[-1])
        predicted_center = [last_center[0] + avg_velocity[0], last_center[1] + avg_velocity[1]]
        
        # Calculate distance between predicted and actual position
        actual_center = self._get_bbox_center(new_bbox)
        distance = np.sqrt(
            (predicted_center[0] - actual_center[0]) ** 2 + 
            (predicted_center[1] - actual_center[1]) ** 2
        )
        
        # Normalize distance (assuming typical movement is within 50 pixels)
        normalized_distance = min(distance / 50.0, 1.0)
        return max(0.0, 1.0 - normalized_distance)
    
    def calculate_motion_consistency(self, new_bbox: List[float]) -> float:
        """Calculate motion consistency based on velocity patterns."""
        if len(self.bbox_history) < 3:
            return 1.0
        
        # Calculate recent velocity
        recent_bboxes = self.bbox_history[-3:]
        velocities = []
        for i in range(1, len(recent_bboxes)):
            prev_center = self._get_bbox_center(recent_bboxes[i-1])
            curr_center = self._get_bbox_center(recent_bboxes[i])
            velocities.append([curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]])
        
        if len(velocities) < 2:
            return 1.0
        
        # Calculate velocity variance (consistency)
        velocity_array = np.array(velocities)
        velocity_variance = np.var(velocity_array, axis=0)
        avg_variance = np.mean(velocity_variance)
        
        # Lower variance = more consistent motion
        consistency_score = max(0.0, 1.0 - (avg_variance / 100.0))  # Normalize by 100
        return consistency_score
    
    def calculate_embedding_consistency(self, new_embedding: np.ndarray) -> float:
        """Calculate embedding consistency with historical embeddings."""
        if not self.embedding_history:
            return 1.0
        
        # Calculate average embedding
        embedding_array = np.array(self.embedding_history)
        avg_embedding = np.mean(embedding_array, axis=0)
        
        # Calculate cosine similarity
        try:
            emb1_norm = new_embedding / np.linalg.norm(new_embedding)
            emb2_norm = avg_embedding / np.linalg.norm(avg_embedding)
            similarity = np.dot(emb1_norm, emb2_norm)
            return max(0.0, similarity)
        except:
            return 0.0
    
    def predict_next_position(self) -> Optional[Dict[str, Any]]:
        """Predict next position based on motion history."""
        if len(self.bbox_history) < 2:
            return None
        
        recent_bboxes = self.bbox_history[-3:]
        if len(recent_bboxes) < 2:
            return None
        
        # Calculate average velocity
        velocities = []
        for i in range(1, len(recent_bboxes)):
            prev_center = self._get_bbox_center(recent_bboxes[i-1])
            curr_center = self._get_bbox_center(recent_bboxes[i])
            velocities.append([curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]])
        
        if not velocities:
            return None
        
        avg_velocity = np.mean(velocities, axis=0)
        last_center = self._get_bbox_center(recent_bboxes[-1])
        predicted_center = [last_center[0] + avg_velocity[0], last_center[1] + avg_velocity[1]]
        
        return {
            'predicted_center': predicted_center,
            'velocity': avg_velocity,
            'confidence': min(1.0, len(velocities) / 5.0)  # More history = higher confidence
        }
    
    def _get_bbox_center(self, bbox: List[float]) -> List[float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]


class StrongSORTTrackingService:
    """StrongSORT tracking service with OSNet ReID for robust person tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize StrongSORT tracking service.
        
        Args:
            config: Tracking configuration
        """
        self.config = config
        self.logger = logger
        
        if not STRONGSORT_AVAILABLE:
            self.logger.error("StrongSORT not available. Install boxmot package.")
            self.tracker = None
            return
        
        # StrongSORT parameters (tuned for better ID stability)
        self.reid_model_path = config.get('reid_model_path', 'models/osnet_x0_25_msmt17.pt')
        self.device = config.get('device', 'cpu')
        self.max_dist = config.get('max_dist', 0.15)  # Further reduced for stricter matching
        self.max_iou_dist = config.get('max_iou_dist', 0.5)  # Further reduced for better IoU matching
        self.max_age = config.get('max_age', 30)  # Reduced for faster track deletion
        self.n_init = config.get('n_init', 5)  # Increased for more stable initialization
        self.nn_budget = config.get('nn_budget', 50)  # Reduced for better performance
        self.mc_lambda = config.get('mc_lambda', 0.995)  # Increased for better motion compensation
        self.ema_alpha = config.get('ema_alpha', 0.9)  # Adjusted for better embedding updates
        self.fp16 = config.get('fp16', False)
        self.per_class = config.get('per_class', False)
        
        # Enhanced tracking features
        self.embedding_smoothing = config.get('embedding_smoothing', {})
        self.detection_stabilization = config.get('detection_stabilization', {})
        
        # Initialize StrongSORT tracker
        self._initialize_tracker()
        
        # Enhanced tracking state
        self.frame_count = 0
        self.track_history = {}
        self.unique_persons = {}
        
        # Rolling average embeddings per track
        self.track_embeddings = {}  # track_id -> list of embeddings
        self.track_embedding_averages = {}  # track_id -> averaged embedding
        
        # Detection stabilization
        self.previous_detections = {}  # track_id -> previous detection
        self.detection_confidence_history = {}  # track_id -> confidence history
        
        # ID stability tracking
        self.id_switch_count = 0
        self.stable_tracks = set()  # Set of stable track IDs
        
        # NEW: Enhanced track reference system to prevent ID switches
        self.track_reference_system = TrackReferenceSystem(config.get('track_reference', {}))
        
        self.logger.info("StrongSORT tracking service initialized with enhanced ID stability")
    
    def _initialize_tracker(self) -> None:
        """Initialize StrongSORT tracker with OSNet ReID."""
        try:
            if not Path(self.reid_model_path).exists():
                self.logger.error(f"ReID model not found: {self.reid_model_path}")
                self.tracker = None
                return
            
            self.tracker = StrongSORT(
                model_weights=Path(self.reid_model_path),
                device=self.device,
                fp16=self.fp16,
                per_class=self.per_class,
                max_dist=self.max_dist,
                max_iou_dist=self.max_iou_dist,
                max_age=self.max_age,
                n_init=self.n_init,
                nn_budget=self.nn_budget,
                mc_lambda=self.mc_lambda,
                ema_alpha=self.ema_alpha
            )
            
            self.logger.info(f"StrongSORT tracker initialized with ReID model: {self.reid_model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize StrongSORT tracker: {e}")
            self.tracker = None
    
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray,
        embeddings: List[Optional[np.ndarray]] = None
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections with enhanced stability and ID switch prevention.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame
            embeddings: Optional face embeddings for ReID
            
        Returns:
            List of tracked objects
        """
        if self.tracker is None:
            return []
        
        self.frame_count += 1
        
        # Apply detection stabilization
        if self.detection_stabilization.get('enabled', False):
            detections = self._stabilize_detections(detections)
        
        # Convert detections to StrongSORT format
        detections_array = self._convert_detections_to_array(detections)
        
        # Update tracker
        try:
            if len(detections_array) == 0:
                outputs = self.tracker.update(np.empty((0, 6)), frame)
            else:
                outputs = self.tracker.update(detections_array, frame)
            
            # Convert outputs to our format
            tracked_objects = self._convert_outputs_to_tracks(outputs)
            
            # Apply embedding smoothing if enabled
            if self.embedding_smoothing.get('enabled', False) and embeddings:
                tracked_objects = self._apply_embedding_smoothing(tracked_objects, embeddings)
            
            # NEW: Apply track reference validation to prevent ID switches
            tracked_objects = self._apply_track_reference_validation(tracked_objects, detections, embeddings)
            
            # Update tracking history and stability
            self._update_tracking_history(tracked_objects)
            self._update_track_stability(tracked_objects)
            
            # Update track reference system
            self._update_track_references(tracked_objects, embeddings)
            
            return tracked_objects
            
        except Exception as e:
            self.logger.error(f"Error updating StrongSORT tracker: {e}")
            return []
    
    def _convert_detections_to_array(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Convert detection format to StrongSORT array format."""
        if not detections:
            return np.empty((0, 6))
        
        detections_array = []
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                confidence = detection.get('confidence', 0.0)
                class_id = detection.get('class_id', 0)
                
                detections_array.append([x1, y1, x2, y2, confidence, class_id])
        
        return np.array(detections_array) if detections_array else np.empty((0, 6))
    
    def _convert_outputs_to_tracks(self, outputs: np.ndarray) -> List[Dict[str, Any]]:
        """Convert StrongSORT outputs to our track format."""
        tracked_objects = []
        
        for output in outputs:
            if len(output) >= 7:  # x1, y1, x2, y2, track_id, conf, class_id
                x1, y1, x2, y2, track_id, conf, class_id = output[:7]
                
                track_info = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'track_id': int(track_id),
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'state': 'confirmed' if conf > 0.5 else 'tentative',
                    'frame_id': self.frame_count,
                    'timestamp': time.time()
                }
                
                tracked_objects.append(track_info)
        
        return tracked_objects
    
    def _update_tracking_history(self, tracked_objects: List[Dict[str, Any]]) -> None:
        """Update tracking history for analysis."""
        for track in tracked_objects:
            track_id = track['track_id']
            
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'first_seen': track['timestamp'],
                    'last_seen': track['timestamp'],
                    'frame_count': 1,
                    'bbox_history': [track['bbox']],
                    'confidence_history': [track['confidence']]
                }
            else:
                history = self.track_history[track_id]
                history['last_seen'] = track['timestamp']
                history['frame_count'] += 1
                history['bbox_history'].append(track['bbox'])
                history['confidence_history'].append(track['confidence'])
                
                # Keep only recent history
                max_history = self.config.get('max_history', 50)
                if len(history['bbox_history']) > max_history:
                    history['bbox_history'] = history['bbox_history'][-max_history:]
                    history['confidence_history'] = history['confidence_history'][-max_history:]
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        if not self.track_history:
            return {'total_tracks': 0, 'active_tracks': 0}
        
        current_time = time.time()
        max_age = self.config.get('track_max_age', 30)  # seconds
        
        active_tracks = 0
        for track_id, history in self.track_history.items():
            if current_time - history['last_seen'] < max_age:
                active_tracks += 1
        
        return {
            'total_tracks': len(self.track_history),
            'active_tracks': active_tracks,
            'frame_count': self.frame_count,
            'tracks': list(self.track_history.keys())
        }
    
    def get_track_by_id(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get track information by ID."""
        if track_id in self.track_history:
            history = self.track_history[track_id]
            return {
                'track_id': track_id,
                'first_seen': history['first_seen'],
                'last_seen': history['last_seen'],
                'frame_count': history['frame_count'],
                'duration': history['last_seen'] - history['first_seen'],
                'avg_confidence': np.mean(history['confidence_history']) if history['confidence_history'] else 0.0
            }
        return None
    
    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Get currently active tracks."""
        current_time = time.time()
        max_age = self.config.get('track_max_age', 30)  # seconds
        
        active_tracks = []
        for track_id, history in self.track_history.items():
            if current_time - history['last_seen'] < max_age:
                active_tracks.append({
                    'track_id': track_id,
                    'first_seen': history['first_seen'],
                    'last_seen': history['last_seen'],
                    'frame_count': history['frame_count'],
                    'duration': history['last_seen'] - history['first_seen']
                })
        
        return active_tracks
    
    def reset_tracker(self) -> None:
        """Reset tracker state."""
        if self.tracker is not None:
            self.tracker.reset()
        
        self.frame_count = 0
        self.track_history = {}
        self.unique_persons = {}
        
        # Reset enhanced tracking state
        self.track_embeddings = {}
        self.track_embedding_averages = {}
        self.previous_detections = {}
        self.detection_confidence_history = {}
        self.id_switch_count = 0
        self.stable_tracks = set()
        
        # Reset track reference system
        self.track_reference_system = TrackReferenceSystem(self.config.get('track_reference', {}))
        
        self.logger.info("StrongSORT tracker reset with enhanced state and track reference system")
    
    def is_available(self) -> bool:
        """Check if StrongSORT is available."""
        return STRONGSORT_AVAILABLE and self.tracker is not None
    
    def _stabilize_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply detection stabilization to reduce jitter and improve tracking stability."""
        stabilized_detections = []
        smoothing_factor = self.detection_stabilization.get('smoothing_factor', 0.7)
        min_confidence = self.detection_stabilization.get('min_confidence', 0.3)
        
        for detection in detections:
            # Filter low confidence detections
            if detection.get('confidence', 0) < min_confidence:
                continue
            
            # Apply smoothing to bounding box if we have previous detection
            bbox = detection['bbox']
            detection_id = detection.get('detection_id', -1)
            
            if detection_id in self.previous_detections:
                prev_bbox = self.previous_detections[detection_id]['bbox']
                
                # Smooth the bounding box coordinates
                smoothed_bbox = [
                    int(smoothing_factor * bbox[0] + (1 - smoothing_factor) * prev_bbox[0]),
                    int(smoothing_factor * bbox[1] + (1 - smoothing_factor) * prev_bbox[1]),
                    int(smoothing_factor * bbox[2] + (1 - smoothing_factor) * prev_bbox[2]),
                    int(smoothing_factor * bbox[3] + (1 - smoothing_factor) * prev_bbox[3])
                ]
                detection['bbox'] = smoothed_bbox
            
            # Store current detection for next frame
            self.previous_detections[detection_id] = detection.copy()
            stabilized_detections.append(detection)
        
        return stabilized_detections
    
    def _apply_embedding_smoothing(self, tracked_objects: List[Dict[str, Any]], 
                                 embeddings: List[Optional[np.ndarray]]) -> List[Dict[str, Any]]:
        """Apply rolling average smoothing to embeddings for better track stability."""
        window_size = self.embedding_smoothing.get('window_size', 5)
        min_embeddings = self.embedding_smoothing.get('min_embeddings', 2)
        
        for i, track in enumerate(tracked_objects):
            track_id = track.get('track_id')
            if track_id is None or i >= len(embeddings) or embeddings[i] is None:
                continue
            
            # Initialize embedding history for new tracks
            if track_id not in self.track_embeddings:
                self.track_embeddings[track_id] = []
            
            # Add current embedding
            self.track_embeddings[track_id].append(embeddings[i])
            
            # Keep only recent embeddings
            if len(self.track_embeddings[track_id]) > window_size:
                self.track_embeddings[track_id] = self.track_embeddings[track_id][-window_size:]
            
            # Calculate rolling average if we have enough embeddings
            if len(self.track_embeddings[track_id]) >= min_embeddings:
                embedding_array = np.array(self.track_embeddings[track_id])
                avg_embedding = np.mean(embedding_array, axis=0)
                self.track_embedding_averages[track_id] = avg_embedding
                
                # Add smoothed embedding to track data
                track['smoothed_embedding'] = avg_embedding
                track['embedding_confidence'] = len(self.track_embeddings[track_id]) / window_size
        
        return tracked_objects
    
    def _update_track_stability(self, tracked_objects: List[Dict[str, Any]]) -> None:
        """Update track stability metrics and detect ID switches."""
        current_track_ids = {track.get('track_id') for track in tracked_objects if track.get('track_id')}
        
        # Detect new stable tracks (tracks that have been active for a while)
        for track in tracked_objects:
            track_id = track.get('track_id')
            if track_id and track_id not in self.stable_tracks:
                # Check if track has been stable for enough frames
                if track_id in self.track_history:
                    history = self.track_history[track_id]
                    if history['frame_count'] >= 10:  # Consider stable after 10 frames
                        self.stable_tracks.add(track_id)
                        self.logger.info(f"Track {track_id} marked as stable")
        
        # Clean up old track data
        self._cleanup_old_tracks(current_track_ids)
    
    def _cleanup_old_tracks(self, current_track_ids: set) -> None:
        """Clean up data for tracks that are no longer active."""
        tracks_to_remove = []
        
        for track_id in self.track_embeddings:
            if track_id not in current_track_ids:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if track_id in self.track_embeddings:
                del self.track_embeddings[track_id]
            if track_id in self.track_embedding_averages:
                del self.track_embedding_averages[track_id]
            if track_id in self.previous_detections:
                del self.previous_detections[track_id]
            if track_id in self.detection_confidence_history:
                del self.detection_confidence_history[track_id]
            if track_id in self.stable_tracks:
                self.stable_tracks.remove(track_id)
    
    def get_track_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Get the smoothed embedding for a track."""
        return self.track_embedding_averages.get(track_id)
    
    def is_track_stable(self, track_id: int) -> bool:
        """Check if a track is considered stable."""
        return track_id in self.stable_tracks
    
    def _apply_track_reference_validation(self, tracked_objects: List[Dict[str, Any]], 
                                        detections: List[Dict[str, Any]], 
                                        embeddings: List[Optional[np.ndarray]]) -> List[Dict[str, Any]]:
        """Apply track reference validation to prevent ID switches."""
        if not tracked_objects:
            return tracked_objects
        
        validated_objects = []
        used_detection_indices = set()
        
        # Sort tracks by confidence and age (prefer stable tracks)
        sorted_tracks = sorted(tracked_objects, key=lambda x: (
            x.get('confidence', 0) * (1 + x.get('frame_id', 0) / 100.0)
        ), reverse=True)
        
        for track in sorted_tracks:
            track_id = track.get('track_id')
            bbox = track.get('bbox', [])
            
            # Find best matching detection for this track
            best_detection_idx = self._find_best_detection_for_track(track, detections, used_detection_indices)
            
            if best_detection_idx is not None:
                detection = detections[best_detection_idx]
                detection_embedding = embeddings[best_detection_idx] if embeddings and best_detection_idx < len(embeddings) else None
                
                # Validate track assignment
                validation_result = self.track_reference_system.validate_track_assignment(
                    track_id, detection['bbox'], detection_embedding
                )
                
                if validation_result['valid']:
                    # Update track with validated detection
                    track['bbox'] = detection['bbox']
                    track['confidence'] = detection['confidence']
                    track['validation_confidence'] = validation_result['confidence']
                    track['validation_reason'] = validation_result['reason']
                    
                    validated_objects.append(track)
                    used_detection_indices.add(best_detection_idx)
                    
                    self.logger.debug(f"Track {track_id} validated: {validation_result['reason']} "
                                    f"(confidence: {validation_result['confidence']:.3f})")
                else:
                    # Track assignment is inconsistent - try to find alternative
                    self.logger.warning(f"Track {track_id} assignment invalid: {validation_result['reason']} "
                                      f"(confidence: {validation_result['confidence']:.3f})")
                    
                    # Check if we can find a better match
                    alternative_track = self._find_alternative_track_assignment(
                        track, detections, used_detection_indices, embeddings
                    )
                    
                    if alternative_track:
                        validated_objects.append(alternative_track)
                        used_detection_indices.add(alternative_track.get('detection_idx'))
                    else:
                        # Keep original track but mark as uncertain
                        track['validation_confidence'] = validation_result['confidence']
                        track['validation_reason'] = 'uncertain'
                        validated_objects.append(track)
            else:
                # No suitable detection found, keep track as is
                track['validation_confidence'] = 0.5
                track['validation_reason'] = 'no_detection'
                validated_objects.append(track)
        
        return validated_objects
    
    def _find_best_detection_for_track(self, track: Dict[str, Any], detections: List[Dict[str, Any]], 
                                     used_indices: set) -> Optional[int]:
        """Find the best detection match for a track."""
        track_bbox = track.get('bbox', [])
        track_id = track.get('track_id')
        
        best_idx = None
        best_score = -1
        
        for i, detection in enumerate(detections):
            if i in used_indices:
                continue
            
            # Calculate IoU score
            iou = self._calculate_iou(track_bbox, detection['bbox'])
            
            # Get track prediction if available
            prediction = self.track_reference_system.get_track_prediction(track_id)
            prediction_score = 0.0
            
            if prediction:
                pred_center = prediction['predicted_center']
                det_center = self._get_bbox_center(detection['bbox'])
                distance = np.sqrt(
                    (pred_center[0] - det_center[0]) ** 2 + 
                    (pred_center[1] - det_center[1]) ** 2
                )
                prediction_score = max(0.0, 1.0 - (distance / 100.0))  # Normalize by 100 pixels
            
            # Combined score
            combined_score = iou * 0.7 + prediction_score * 0.3
            
            if combined_score > best_score and combined_score > 0.3:  # Minimum threshold
                best_score = combined_score
                best_idx = i
        
        return best_idx
    
    def _find_alternative_track_assignment(self, track: Dict[str, Any], detections: List[Dict[str, Any]], 
                                         used_indices: set, embeddings: List[Optional[np.ndarray]]) -> Optional[Dict[str, Any]]:
        """Find alternative track assignment when current one is invalid."""
        track_id = track.get('track_id')
        
        # Look for other tracks that might be better matches
        for i, detection in enumerate(detections):
            if i in used_indices:
                continue
            
            detection_embedding = embeddings[i] if embeddings and i < len(embeddings) else None
            
            # Check if this detection would be a better match for this track
            validation_result = self.track_reference_system.validate_track_assignment(
                track_id, detection['bbox'], detection_embedding
            )
            
            if validation_result['valid'] and validation_result['confidence'] > 0.7:
                # Create alternative track
                alternative_track = track.copy()
                alternative_track['bbox'] = detection['bbox']
                alternative_track['confidence'] = detection['confidence']
                alternative_track['validation_confidence'] = validation_result['confidence']
                alternative_track['validation_reason'] = 'alternative_match'
                alternative_track['detection_idx'] = i
                
                return alternative_track
        
        return None
    
    def _update_track_references(self, tracked_objects: List[Dict[str, Any]], 
                               embeddings: List[Optional[np.ndarray]]) -> None:
        """Update track reference system with current frame data."""
        active_track_ids = set()
        
        for i, track in enumerate(tracked_objects):
            track_id = track.get('track_id')
            if track_id is None:
                continue
            
            active_track_ids.add(track_id)
            
            bbox = track.get('bbox', [])
            confidence = track.get('confidence', 1.0)
            embedding = embeddings[i] if embeddings and i < len(embeddings) else None
            
            # Update track reference
            self.track_reference_system.update_track_reference(
                track_id, bbox, embedding, confidence
            )
        
        # Clean up old references
        self.track_reference_system.cleanup_old_references(active_track_ids)
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_bbox_center(self, bbox: List[float]) -> List[float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
