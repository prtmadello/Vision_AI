"""
Lightweight ReID tracking service for Vision AI system.
Optimized for computational efficiency while maintaining reasonable tracking quality.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import json
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class LightweightTrack:
    """Lightweight track representation"""
    track_id: int
    bbox: List[float]
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    state: str = "Tentative"
    last_seen: float = 0.0
    velocity: List[float] = None
    bbox_history: deque = None
    max_history: int = 5
    consecutive_misses: int = 0
    simple_features: Optional[np.ndarray] = None  # Simple color/texture features
    feature_history: deque = None  # Store multiple feature vectors for better matching
    max_feature_history: int = 5
    
    def __post_init__(self):
        if self.bbox_history is None:
            self.bbox_history = deque(maxlen=self.max_history)
            self.bbox_history.append(self.bbox.copy())
        if self.velocity is None:
            self.velocity = [0.0, 0.0]
        if self.feature_history is None:
            self.feature_history = deque(maxlen=self.max_feature_history)


class LightweightTrackingService:
    """Lightweight tracking service optimized for computational efficiency."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize lightweight tracking service.
        
        Args:
            config: Tracking configuration
        """
        self.config = config
        self.logger = logger
        
        # Core tracking parameters
        self.max_disappeared = config.get('max_disappeared', 15)
        self.max_distance = config.get('max_distance', 0.4)
        self.min_hits = config.get('min_hits', 3)
        self.max_age = config.get('max_age', 1)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.motion_weight = config.get('motion_weight', 0.3)
        self.feature_weight = config.get('feature_weight', 0.2)
        self.iou_weight = config.get('iou_weight', 0.5)
        self.max_consecutive_misses = config.get('max_consecutive_misses', 5)
        
        # Feature extraction parameters
        self.use_simple_features = config.get('use_simple_features', True)
        self.feature_extraction_interval = config.get('feature_extraction_interval', 3)  # Extract features every N frames
        
        # Motion prediction parameters
        self.use_motion_prediction = config.get('use_motion_prediction', True)
        self.motion_smoothing_factor = config.get('motion_smoothing_factor', 0.7)
        self.max_prediction_distance = config.get('max_prediction_distance', 100.0)
        
        # Performance optimization
        self.enable_parallel_processing = config.get('enable_parallel_processing', False)
        self.max_tracks = config.get('max_tracks', 50)  # Limit number of active tracks
        
        # Tracking state
        self.tracks: List[LightweightTrack] = []
        self.next_id = 1
        self.frame_count = 0
        
        # Persistent feature storage for better ReID
        self.persistent_features: Dict[int, List[np.ndarray]] = {}  # track_id -> list of feature vectors
        self.feature_storage_path = Path("output/tracking_features.pkl")
        self.load_persistent_features()
        
        # Improved assignment parameters
        self.feature_similarity_threshold = config.get('feature_similarity_threshold', 0.7)
        self.min_feature_matches = config.get('min_feature_matches', 2)
        self.use_hungarian_assignment = config.get('use_hungarian_assignment', True)
        
        # Performance metrics
        self.processing_times = deque(maxlen=100)
        self.feature_extraction_times = deque(maxlen=100)
        
        self.logger.info("Lightweight tracking service initialized with optimized parameters and persistent features")
    
    def load_persistent_features(self) -> None:
        """Load persistent features from storage."""
        try:
            if self.feature_storage_path.exists():
                with open(self.feature_storage_path, 'rb') as f:
                    self.persistent_features = pickle.load(f)
                self.logger.info(f"Loaded persistent features for {len(self.persistent_features)} tracks")
            else:
                self.persistent_features = {}
                self.logger.info("No persistent features found, starting fresh")
        except Exception as e:
            self.logger.warning(f"Failed to load persistent features: {e}")
            self.persistent_features = {}
    
    def save_persistent_features(self) -> None:
        """Save persistent features to storage."""
        try:
            self.feature_storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.feature_storage_path, 'wb') as f:
                pickle.dump(self.persistent_features, f)
            self.logger.debug(f"Saved persistent features for {len(self.persistent_features)} tracks")
        except Exception as e:
            self.logger.warning(f"Failed to save persistent features: {e}")
    
    def add_persistent_feature(self, track_id: int, feature: np.ndarray) -> None:
        """Add a feature to persistent storage for a track - keep only latest."""
        # Keep only the latest feature for simplicity
        self.persistent_features[track_id] = [feature.copy()]
    
    def find_best_feature_match(self, detection_feature: np.ndarray, exclude_track_ids: set = None) -> Optional[Tuple[int, float]]:
        """Find the best matching track based on persistent features."""
        if exclude_track_ids is None:
            exclude_track_ids = set()
        
        best_match_id = None
        best_similarity = 0.0
        
        for track_id, features in self.persistent_features.items():
            if track_id in exclude_track_ids:
                continue
            
            if not features:
                continue
            
            # Calculate similarity with all stored features for this track
            similarities = []
            for stored_feature in features:
                similarity = 1.0 - self._calculate_feature_distance(detection_feature, stored_feature)
                similarities.append(similarity)
            
            # Use the maximum similarity as the track similarity
            track_similarity = max(similarities) if similarities else 0.0
            
            if track_similarity > best_similarity and track_similarity > self.feature_similarity_threshold:
                best_similarity = track_similarity
                best_match_id = track_id
        
        return (best_match_id, best_similarity) if best_match_id is not None else None
    
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray = None
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame (optional, for feature extraction)
            
        Returns:
            List of tracked objects
        """
        start_time = time.time()
        self.frame_count += 1
        
        if not detections:
            # Update existing tracks
            self._update_track_ages()
            return self._get_tracked_objects()
        
        # Update track ages and states
        self._update_track_ages()
        
        # Skip frames for faster processing (process every 10th frame)
        if self.frame_count % 10 != 0:
            # Just update existing tracks without new detections
            self._update_track_ages()
            return self._get_tracked_objects()
        
        # Extract simple features if enabled and frame is available
        detection_features = None
        if self.use_simple_features and frame is not None:
            detection_features = self._extract_simple_features(detections, frame)
        
        # Associate detections with tracks
        matched_tracks, unmatched_detections, unmatched_tracks = self._associate_detections(
            detections, detection_features
        )
        
        # Update matched tracks
        self._update_matched_tracks(matched_tracks, detections, detection_features)
        
        # Create new tracks for unmatched detections
        self._create_new_tracks(unmatched_detections, detections, detection_features)
        
        # Remove old tracks
        self._remove_old_tracks()
        
        # Limit number of tracks for performance
        self._limit_tracks()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Save persistent features periodically (every 100 frames)
        if self.frame_count % 100 == 0:
            self.save_persistent_features()
        
        return self._get_tracked_objects()
    
    def _update_track_ages(self) -> None:
        """Update age and time since update for all tracks."""
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1
            track.consecutive_misses += 1
    
    def _extract_simple_features(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Optional[np.ndarray]]:
        """Extract simple color and texture features from detections."""
        start_time = time.time()
        features = []
        
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) != 4:
                features.append(None)
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                features.append(None)
                continue
            
            try:
                # Extract region of interest
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    features.append(None)
                    continue
                
                # Simple feature extraction
                feature_vector = self._compute_simple_features(roi)
                features.append(feature_vector)
                
            except Exception as e:
                self.logger.debug(f"Feature extraction failed: {e}")
                features.append(None)
        
        feature_time = time.time() - start_time
        self.feature_extraction_times.append(feature_time)
        
        return features
    
    def _compute_simple_features(self, roi: np.ndarray) -> np.ndarray:
        """Compute simple color and texture features from ROI."""
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) if len(roi.shape) == 3 else None
            
            features = []
            
            # Color features (mean and std of each channel)
            if hsv is not None:
                for channel in cv2.split(hsv):
                    features.extend([np.mean(channel), np.std(channel)])
            else:
                features.extend([np.mean(gray), np.std(gray)])
            
            # Texture features (simple gradient statistics)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.percentile(gradient_magnitude, 25),
                np.percentile(gradient_magnitude, 75)
            ])
            
            # Size and aspect ratio
            h, w = roi.shape[:2]
            features.extend([w, h, w/h])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.debug(f"Feature computation failed: {e}")
            return np.zeros(10, dtype=np.float32)  # Return default features
    
    def _associate_detections(
        self,
        detections: List[Dict[str, Any]],
        detection_features: List[Optional[np.ndarray]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections with existing tracks using lightweight matching."""
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Build cost matrix
        cost_matrix = np.full((len(self.tracks), len(detections)), np.inf)
        
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                # Calculate IoU cost
                iou = self._calculate_iou(track.bbox, detection['bbox'])
                iou_cost = 1.0 - iou
                
                if iou < self.iou_threshold:
                    continue
                
                # Calculate motion cost
                motion_cost = 0.0
                if self.use_motion_prediction and track.velocity is not None:
                    motion_cost = self._calculate_motion_cost(track, detection['bbox'])
                
                # Calculate feature cost
                feature_cost = 0.0
                if (self.use_simple_features and 
                    track.simple_features is not None and 
                    detection_features and 
                    detection_features[j] is not None):
                    feature_cost = self._calculate_feature_distance(
                        track.simple_features, detection_features[j]
                    )
                
                # Combined cost with more aggressive IoU weighting to prevent ID flips
                total_cost = (
                    0.8 * iou_cost +  # Increased IoU weight to prevent flips
                    self.motion_weight * motion_cost +
                    self.feature_weight * feature_cost
                )
                
                # Be more lenient with distance threshold to prevent ID flips
                if total_cost < self.max_distance and iou > 0.15:  # Minimum IoU threshold
                    cost_matrix[i, j] = total_cost
        
        # Use Hungarian algorithm for better assignment if enabled and matrix is small enough
        if (self.use_hungarian_assignment and 
            len(self.tracks) <= 10 and len(detections) <= 10 and
            not np.all(np.isinf(cost_matrix))):
            return self._hungarian_assignment(cost_matrix)
        else:
            return self._greedy_assignment(cost_matrix)
    
    def _greedy_assignment(self, cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Simple greedy assignment for efficiency."""
        matched_pairs = []
        used_tracks = set()
        used_detections = set()
        
        # Create list of all possible assignments sorted by cost
        assignments = []
        for i in range(len(self.tracks)):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] < self.max_distance:
                    assignments.append((cost_matrix[i, j], i, j))
        
        # Sort by cost (best matches first)
        assignments.sort()
        
        # Greedily assign best matches
        for cost, i, j in assignments:
            if i not in used_tracks and j not in used_detections:
                matched_pairs.append((i, j))
                used_tracks.add(i)
                used_detections.add(j)
        
        # Find unmatched tracks and detections
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        unmatched_detections = [j for j in range(cost_matrix.shape[1]) if j not in used_detections]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _hungarian_assignment(self, cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian algorithm assignment for optimal matching."""
        try:
            from scipy.optimize import linear_sum_assignment
            
            # Apply Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            matched_pairs = []
            used_tracks = set()
            used_detections = set()
            
            for i, j in zip(row_indices, col_indices):
                if cost_matrix[i, j] < self.max_distance:
                    matched_pairs.append((i, j))
                    used_tracks.add(i)
                    used_detections.add(j)
            
            # Find unmatched tracks and detections
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
            unmatched_detections = [j for j in range(cost_matrix.shape[1]) if j not in used_detections]
            
            return matched_pairs, unmatched_detections, unmatched_tracks
            
        except ImportError:
            self.logger.warning("scipy not available, falling back to greedy assignment")
            return self._greedy_assignment(cost_matrix)
        except Exception as e:
            self.logger.warning(f"Hungarian assignment failed: {e}, falling back to greedy")
            return self._greedy_assignment(cost_matrix)
    
    def _update_matched_tracks(
        self,
        matched_tracks: List[Tuple[int, int]],
        detections: List[Dict[str, Any]],
        detection_features: List[Optional[np.ndarray]]
    ) -> None:
        """Update tracks that were matched with detections."""
        for track_idx, det_idx in matched_tracks:
            track = self.tracks[track_idx]
            detection = detections[det_idx]
            
            # Update track properties
            old_bbox = track.bbox.copy()
            track.bbox = detection['bbox']
            track.confidence = detection['confidence']
            track.hits += 1
            track.time_since_update = 0
            track.consecutive_misses = 0
            track.last_seen = time.time()
            
            # Update bbox history
            track.bbox_history.append(detection['bbox'].copy())
            
            # Update velocity
            if self.use_motion_prediction and len(track.bbox_history) >= 2:
                self._update_track_velocity(track, old_bbox, detection['bbox'])
            
            # Update features
            if (self.use_simple_features and 
                detection_features and 
                detection_features[det_idx] is not None):
                track.simple_features = detection_features[det_idx]
                track.feature_history.append(detection_features[det_idx])
                
                # Add to persistent storage less frequently to avoid overhead
                if track.hits % 10 == 0:  # Store every 10th hit
                    self.add_persistent_feature(track.track_id, detection_features[det_idx])
            
            # Update track state
            if track.hits >= self.min_hits:
                track.state = "Confirmed"
    
    def _create_new_tracks(
        self,
        unmatched_detections: List[int],
        detections: List[Dict[str, Any]],
        detection_features: List[Optional[np.ndarray]]
    ) -> None:
        """Create new tracks for unmatched detections with simple feature matching."""
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            detection_feature = detection_features[det_idx] if detection_features else None
            
            # Simple feature comparison to find existing person
            matched_track_id = None
            if detection_feature is not None and self.persistent_features:
                best_similarity = 0.0
                for stored_track_id, stored_features in self.persistent_features.items():
                    if not stored_features:
                        continue
                    
                    # Compare with the most recent feature
                    latest_feature = stored_features[-1]
                    similarity = 1.0 - self._calculate_feature_distance(detection_feature, latest_feature)
                    
                    if similarity > best_similarity and similarity > 0.6:  # Simple threshold
                        best_similarity = similarity
                        matched_track_id = stored_track_id
                
                if matched_track_id is not None:
                    self.logger.info(f"Re-identified person with track ID {matched_track_id} (similarity: {best_similarity:.3f})")
            
            # Create new track
            if matched_track_id is not None:
                track_id = matched_track_id
            else:
                track_id = self.next_id
                self.next_id += 1
            
            new_track = LightweightTrack(
                track_id=track_id,
                bbox=detection['bbox'],
                confidence=detection['confidence'],
                simple_features=detection_feature
            )
            
            # Store feature for future comparison
            if detection_feature is not None:
                self.add_persistent_feature(track_id, detection_feature)
                new_track.feature_history.append(detection_feature)
            
            self.tracks.append(new_track)
    
    def _remove_old_tracks(self) -> None:
        """Remove tracks that have been missing for too long."""
        self.tracks = [
            track for track in self.tracks
            if (track.time_since_update < self.max_disappeared and 
                track.consecutive_misses < self.max_consecutive_misses)
        ]
    
    def _limit_tracks(self) -> None:
        """Limit number of tracks for performance."""
        if len(self.tracks) > self.max_tracks:
            # Keep tracks with highest confidence and most hits
            self.tracks.sort(key=lambda t: (t.confidence * t.hits), reverse=True)
            self.tracks = self.tracks[:self.max_tracks]
    
    def _update_track_velocity(self, track: LightweightTrack, old_bbox: List[float], new_bbox: List[float]) -> None:
        """Update track velocity using exponential smoothing."""
        old_center = self._get_bbox_center(old_bbox)
        new_center = self._get_bbox_center(new_bbox)
        
        current_velocity = [new_center[0] - old_center[0], new_center[1] - old_center[1]]
        
        if track.velocity is None:
            track.velocity = current_velocity
        else:
            # Exponential smoothing
            track.velocity[0] = (self.motion_smoothing_factor * current_velocity[0] + 
                               (1 - self.motion_smoothing_factor) * track.velocity[0])
            track.velocity[1] = (self.motion_smoothing_factor * current_velocity[1] + 
                               (1 - self.motion_smoothing_factor) * track.velocity[1])
    
    def _calculate_motion_cost(self, track: LightweightTrack, detection_bbox: List[float]) -> float:
        """Calculate motion cost based on predicted position."""
        if track.velocity is None or len(track.bbox_history) == 0:
            return 0.0
        
        # Predict next position
        last_center = self._get_bbox_center(track.bbox_history[-1])
        predicted_center = [
            last_center[0] + track.velocity[0],
            last_center[1] + track.velocity[1]
        ]
        
        # Calculate distance to actual detection
        actual_center = self._get_bbox_center(detection_bbox)
        distance = np.sqrt(
            (predicted_center[0] - actual_center[0]) ** 2 + 
            (predicted_center[1] - actual_center[1]) ** 2
        )
        
        # Normalize distance
        normalized_distance = min(distance / self.max_prediction_distance, 1.0)
        return normalized_distance
    
    def _calculate_feature_distance(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate distance between feature vectors."""
        try:
            # Normalize features
            f1_norm = features1 / (np.linalg.norm(features1) + 1e-8)
            f2_norm = features2 / (np.linalg.norm(features2) + 1e-8)
            
            # Calculate cosine distance
            similarity = np.dot(f1_norm, f2_norm)
            distance = 1.0 - similarity
            return max(0.0, distance)
        except:
            return 1.0
    
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
    
    def _get_tracked_objects(self) -> List[Dict[str, Any]]:
        """Get list of tracked objects in the required format."""
        tracked_objects = []
        
        for track in self.tracks:
            if track.state == "Confirmed" or track.hits >= self.min_hits:
                tracked_object = {
                    'bbox': track.bbox,
                    'track_id': track.track_id,
                    'confidence': track.confidence,
                    'class_id': 0,  # Assuming person class
                    'state': track.state,
                    'frame_id': self.frame_count,
                    'timestamp': time.time(),
                    'age': track.age,
                    'hits': track.hits,
                    'velocity': track.velocity
                }
                tracked_objects.append(tracked_object)
        
        return tracked_objects
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics and performance metrics."""
        confirmed_tracks = [t for t in self.tracks if t.state == "Confirmed"]
        tentative_tracks = [t for t in self.tracks if t.state == "Tentative"]
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        avg_feature_time = np.mean(self.feature_extraction_times) if self.feature_extraction_times else 0.0
        
        return {
            'total_tracks': len(self.tracks),
            'confirmed_tracks': len(confirmed_tracks),
            'tentative_tracks': len(tentative_tracks),
            'next_id': self.next_id,
            'frame_count': self.frame_count,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'avg_feature_extraction_time_ms': avg_feature_time * 1000,
            'avg_track_age': np.mean([t.age for t in self.tracks]) if self.tracks else 0,
            'avg_track_hits': np.mean([t.hits for t in self.tracks]) if self.tracks else 0
        }
    
    def get_track_by_id(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get track information by ID."""
        for track in self.tracks:
            if track.track_id == track_id:
                return {
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'age': track.age,
                    'hits': track.hits,
                    'state': track.state,
                    'last_seen': track.last_seen,
                    'velocity': track.velocity
                }
        return None
    
    def reset_tracker(self) -> None:
        """Reset tracker state."""
        # Save persistent features before reset
        self.save_persistent_features()
        
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.processing_times.clear()
        self.feature_extraction_times.clear()
        self.logger.info("Lightweight tracker reset")
    
    def is_available(self) -> bool:
        """Check if lightweight tracker is available."""
        return True  # Always available as it doesn't depend on external models


