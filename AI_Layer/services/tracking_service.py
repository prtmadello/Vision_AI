"""
Face tracking service for AI Layer
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Install scipy for optimal tracking performance.")

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Track:
    """Represents a tracked face"""
    track_id: int
    bbox: List[float]
    confidence: float
    embedding: Optional[np.ndarray] = None
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    state: str = "Tentative"
    last_seen: float = 0.0
    bbox_history: List[List[float]] = None
    max_history: int = 5
    consecutive_misses: int = 0
    
    def __post_init__(self):
        if self.bbox_history is None:
            self.bbox_history = [self.bbox.copy()]


class FaceTrackingService:
    """Face tracking service using DeepSORT algorithm"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        
        self.max_disappeared = config.get('max_disappeared', 10)
        self.max_distance = config.get('max_distance', 0.4)
        self.min_hits = config.get('min_hits', 3)
        self.max_age = config.get('max_age', 1)
        self.embedding_threshold = config.get('embedding_threshold', 0.6)
        self.iou_threshold = config.get('iou_threshold', 0.5)
        self.motion_weight = config.get('motion_weight', 0.2)
        self.embedding_weight = config.get('embedding_weight', 0.8)
        self.max_consecutive_misses = config.get('max_consecutive_misses', 3)
        
        self.tracks: List[Track] = []
        self.next_id = 1
        
        self.logger.info("FaceTrackingService initialized")
    
    def update(
        self,
        detections: List[Dict[str, Any]],
        embeddings: List[Optional[np.ndarray]] = None
    ) -> List[Dict[str, Any]]:
        """Update tracks with new detections"""
        if embeddings is None:
            embeddings = [None] * len(detections)
        
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1
            track.consecutive_misses += 1
        
        matched_tracks, unmatched_detections, unmatched_tracks = self._associate_detections(
            detections, embeddings
        )
        
        for track_idx, det_idx in matched_tracks:
            track = self.tracks[track_idx]
            detection = detections[det_idx]
            embedding = embeddings[det_idx]
            
            track.bbox = detection['bbox']
            track.confidence = detection['confidence']
            track.embedding = embedding
            track.hits += 1
            track.time_since_update = 0
            track.consecutive_misses = 0
            
            track.bbox_history.append(detection['bbox'].copy())
            if len(track.bbox_history) > track.max_history:
                track.bbox_history.pop(0)
            
            if track.hits >= self.min_hits:
                track.state = "Confirmed"
        
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            embedding = embeddings[det_idx]
            
            new_track = Track(
                track_id=self.next_id,
                bbox=detection['bbox'],
                confidence=detection['confidence'],
                embedding=embedding
            )
            
            self.tracks.append(new_track)
            self.next_id += 1
        
        self.tracks = [
            track for track in self.tracks
            if (track.time_since_update < self.max_disappeared and 
                track.consecutive_misses < self.max_consecutive_misses)
        ]
        
        tracked_faces = []
        for track in self.tracks:
            if track.state == "Confirmed":
                tracked_face = {
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'embedding': track.embedding,
                    'age': track.age,
                    'hits': track.hits,
                    'state': track.state
                }
                tracked_faces.append(tracked_face)
        
        return tracked_faces
    
    def _associate_detections(
        self,
        detections: List[Dict[str, Any]],
        embeddings: List[Optional[np.ndarray]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections with existing tracks"""
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        cost_matrix = np.full((len(self.tracks), len(detections)), np.inf)
        
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                iou = self._calculate_iou(track.bbox, detection['bbox'])
                iou_cost = 1.0 - iou
                
                embedding_cost = 0.0
                if track.embedding is not None and embeddings[j] is not None:
                    embedding_cost = self._calculate_embedding_distance(
                        track.embedding, embeddings[j]
                    )
                
                motion_cost = self._calculate_motion_cost(track, detection['bbox'])
                
                if iou > self.iou_threshold:
                    if embedding_cost > 0:
                        total_cost = (
                            (1.0 - self.motion_weight - self.embedding_weight) * iou_cost +
                            self.embedding_weight * embedding_cost +
                            self.motion_weight * motion_cost
                        )
                        cost_matrix[i, j] = total_cost
                    else:
                        total_cost = (1.0 - self.motion_weight) * iou_cost + self.motion_weight * motion_cost
                        cost_matrix[i, j] = total_cost
        
        try:
            if SCIPY_AVAILABLE and cost_matrix.size > 0:
                if np.all(np.isinf(cost_matrix)) or cost_matrix.size == 0:
                    return [], list(range(len(detections))), list(range(len(self.tracks)))
                
                track_indices, det_indices = linear_sum_assignment(cost_matrix)
                
                matched_pairs = []
                for t_idx, d_idx in zip(track_indices, det_indices):
                    if cost_matrix[t_idx, d_idx] < self.max_distance:
                        matched_pairs.append((t_idx, d_idx))
            else:
                matched_pairs = self._greedy_assignment(cost_matrix)
        except Exception as e:
            self.logger.debug(f"Assignment failed: {e}, using greedy assignment")
            matched_pairs = self._greedy_assignment(cost_matrix)
        
        matched_tracks = {pair[0] for pair in matched_pairs}
        matched_detections = {pair[1] for pair in matched_pairs}
        
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        unmatched_detections = [j for j in range(len(detections)) if j not in matched_detections]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _greedy_assignment(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Simple greedy assignment when scipy is not available"""
        matched_pairs = []
        used_tracks = set()
        used_detections = set()
        
        if cost_matrix.size == 0 or len(self.tracks) == 0:
            return matched_pairs
        
        assignments = []
        for i in range(len(self.tracks)):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] < self.max_distance:
                    assignments.append((cost_matrix[i, j], i, j))
        
        assignments.sort()
        
        for cost, i, j in assignments:
            if i not in used_tracks and j not in used_detections:
                matched_pairs.append((i, j))
                used_tracks.add(i)
                used_detections.add(j)
        
        return matched_pairs
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
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
    
    def _calculate_embedding_distance(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine distance between embeddings"""
        try:
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            similarity = np.dot(emb1_norm, emb2_norm)
            distance = 1.0 - similarity
            return distance
        except:
            return 1.0
    
    def _calculate_motion_cost(self, track: Track, detection_bbox: List[float]) -> float:
        """Calculate motion cost based on track history"""
        if len(track.bbox_history) < 2:
            return 0.0
        
        try:
            recent_bboxes = track.bbox_history[-3:]
            if len(recent_bboxes) < 2:
                return 0.0
            
            velocities = []
            for i in range(1, len(recent_bboxes)):
                prev_bbox = recent_bboxes[i-1]
                curr_bbox = recent_bboxes[i]
                
                prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
                prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
                curr_cx = (curr_bbox[0] + curr_bbox[2]) / 2
                curr_cy = (curr_bbox[1] + curr_bbox[3]) / 2
                
                velocities.append([curr_cx - prev_cx, curr_cy - prev_cy])
            
            if not velocities:
                return 0.0
            
            avg_velocity = np.mean(velocities, axis=0)
            last_bbox = recent_bboxes[-1]
            last_cx = (last_bbox[0] + last_bbox[2]) / 2
            last_cy = (last_bbox[1] + last_bbox[3]) / 2
            
            predicted_cx = last_cx + avg_velocity[0]
            predicted_cy = last_cy + avg_velocity[1]
            
            detection_cx = (detection_bbox[0] + detection_bbox[2]) / 2
            detection_cy = (detection_bbox[1] + detection_bbox[3]) / 2
            
            distance = np.sqrt(
                (predicted_cx - detection_cx) ** 2 + 
                (predicted_cy - detection_cy) ** 2
            )
            
            normalized_distance = min(distance / 100.0, 1.0)
            return normalized_distance
            
        except Exception as e:
            self.logger.debug(f"Motion cost calculation failed: {e}")
            return 0.0
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        confirmed_tracks = [t for t in self.tracks if t.state == "Confirmed"]
        tentative_tracks = [t for t in self.tracks if t.state == "Tentative"]
        
        return {
            'total_tracks': len(self.tracks),
            'confirmed_tracks': len(confirmed_tracks),
            'tentative_tracks': len(tentative_tracks),
            'next_id': self.next_id,
            'avg_track_age': np.mean([t.age for t in self.tracks]) if self.tracks else 0,
            'avg_track_hits': np.mean([t.hits for t in self.tracks]) if self.tracks else 0
        }
    
    def reset(self):
        """Reset tracking service"""
        self.tracks = []
        self.next_id = 1
        self.logger.info("Tracking service reset")
