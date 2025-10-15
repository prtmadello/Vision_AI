"""
Face tracking service for AI Layer
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

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
        
        # Enhanced persistent person re-identification system
        self.persistent_tracks = {}  # track_id -> Track
        self.lost_tracks = {}  # track_id -> {'last_seen': frame, 'embedding': np.array, 'bbox': list}
        self.person_embeddings = {}  # track_id -> list of embeddings
        self.next_persistent_id = 1
        self.persistent_id_mapping = {}  # track_id -> persistent_id
        self.display_id_mapping = {}  # persistent_id -> display_id (for UI consistency)
        self.next_display_id = 1
        self.frame_count = 0
        
        # Configuration for persistent re-identification
        self.persistent_reid_enabled = config.get('persistent_reid', {}).get('enabled', True)
        self.persistent_similarity_threshold = config.get('persistent_reid', {}).get('similarity_threshold', 0.7)
        self.max_track_id_age = config.get('persistent_reid', {}).get('max_age', 1000)  # frames
        
        # JSON storage configuration
        self.embeddings_json_path = config.get('persistent_reid', {}).get('embeddings_json_path', 'data/person_embeddings.json')
        self.person_details_json_path = config.get('persistent_reid', {}).get('person_details_json_path', 'data/person_details.json')
        
        # Load existing embeddings and person details
        self._load_persistent_data()
        
        # Multiprocessing configuration
        self.use_multiprocessing = config.get('multiprocessing', {}).get('enabled', True)
        self.max_workers = config.get('multiprocessing', {}).get('max_workers', min(4, cpu_count()))
        
        self.logger.info(f"FaceTrackingService initialized with persistent ReID: {self.persistent_reid_enabled}")
        self.logger.info(f"Multiprocessing enabled: {self.use_multiprocessing} (workers: {self.max_workers})")
        self.logger.info(f"JSON storage paths - Embeddings: {self.embeddings_json_path}, Details: {self.person_details_json_path}")
    
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Update tracks with new detections"""
        self.frame_count += 1
        
        # Extract person embeddings from frame if available
        embeddings = []
        if frame is not None:
            self.logger.info(f"Extracting embeddings for {len(detections)} detections")
            if self.use_multiprocessing and len(detections) > 1:
                embeddings = self._extract_embeddings_parallel(frame, detections)
            else:
                for i, detection in enumerate(detections):
                    embedding = self._extract_person_embedding(frame, detection['bbox'])
                    embeddings.append(embedding)
                    if embedding is not None:
                        self.logger.info(f"Extracted embedding for detection {i}: shape {embedding.shape}")
                    else:
                        self.logger.warning(f"Failed to extract embedding for detection {i}")
        else:
            embeddings = [None] * len(detections)
            self.logger.warning("No frame provided, skipping embedding extraction")
        
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1
            track.consecutive_misses += 1
        
        matched_tracks, unmatched_detections, unmatched_tracks = self._associate_detections(
            detections, embeddings
        )
        
        self.logger.debug(f"Association results - Matched: {len(matched_tracks)}, Unmatched detections: {len(unmatched_detections)}, Unmatched tracks: {len(unmatched_tracks)}")
        
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
            
            # Try to find a matching lost track BEFORE creating new track
            self.logger.debug(f"Looking for match for new detection: embedding={embedding is not None}, lost_tracks={len(self.lost_tracks)}")
            persistent_id = self._find_matching_lost_track(embedding)
            if persistent_id is not None:
                self.logger.info(f"Found re-identification match: persistent_id={persistent_id}")
            else:
                # If no match found, try to find a match with existing active tracks (for same person with different head position)
                persistent_id = self._find_matching_active_track(embedding)
                if persistent_id is not None:
                    self.logger.info(f"Found match with active track: persistent_id={persistent_id}")
                else:
                    self.logger.info(f"No re-identification match found, will create new persistent ID")
            
            new_track = Track(
                track_id=self.next_id,
                bbox=detection['bbox'],
                confidence=detection['confidence'],
                embedding=embedding
            )
            
            self.tracks.append(new_track)
            
            # Assign persistent ID (either reused or new)
            if persistent_id is not None:
                # Check if this persistent ID is already used by ACTIVE tracks only
                active_persistent_ids = set()
                for track in self.tracks:
                    if track.track_id in self.persistent_id_mapping:
                        active_persistent_ids.add(self.persistent_id_mapping[track.track_id])
                
                if persistent_id not in active_persistent_ids:
                    self.persistent_id_mapping[new_track.track_id] = persistent_id
                    
                    # Remove the matched lost track to prevent duplicate matches
                    for lost_track_id, lost_data in list(self.lost_tracks.items()):
                        if lost_data.get('persistent_id') == persistent_id:
                            del self.lost_tracks[lost_track_id]
                            break
                    
                    self.logger.info(f"Re-identified person: Track {new_track.track_id} -> Persistent ID {persistent_id}")
                else:
                    # Persistent ID is already assigned to active track, assign new one
                    new_persistent_id = self.next_persistent_id
                    self.next_persistent_id += 1
                    self.persistent_id_mapping[new_track.track_id] = new_persistent_id
                    self.logger.warning(f"Persistent ID {persistent_id} already in use by active track, assigned new ID {new_persistent_id} to Track {new_track.track_id}")
            else:
                # Check if we already have too many persistent IDs (limit to reasonable number)
                if len(self.persistent_id_mapping) > 10:  # Limit to 10 unique people
                    self.logger.warning(f"Too many persistent IDs ({len(self.persistent_id_mapping)}), trying to find closest match")
                    # Try to find the closest existing persistent ID
                    if embedding is not None:
                        closest_id = self._find_closest_existing_persistent_id(embedding)
                        if closest_id is not None:
                            self.persistent_id_mapping[new_track.track_id] = closest_id
                            self.logger.info(f"Assigned closest existing persistent ID {closest_id} to Track {new_track.track_id}")
                        else:
                            # Assign new persistent ID
                            new_persistent_id = self.next_persistent_id
                            self.next_persistent_id += 1
                            self.persistent_id_mapping[new_track.track_id] = new_persistent_id
                            self.logger.info(f"New person detected: Track {new_track.track_id} -> Persistent ID {new_persistent_id}")
                    else:
                        # Assign new persistent ID
                        new_persistent_id = self.next_persistent_id
                        self.next_persistent_id += 1
                        self.persistent_id_mapping[new_track.track_id] = new_persistent_id
                        self.logger.info(f"New person detected: Track {new_track.track_id} -> Persistent ID {new_persistent_id}")
                else:
                    # Assign new persistent ID
                    new_persistent_id = self.next_persistent_id
                    self.next_persistent_id += 1
                    self.persistent_id_mapping[new_track.track_id] = new_persistent_id
                    self.logger.info(f"New person detected: Track {new_track.track_id} -> Persistent ID {new_persistent_id}")
            
            self.next_id += 1
        
        # Update frame count
        self.frame_count += 1
        
        # Store lost tracks before removing them
        for track in self.tracks:
            if (track.time_since_update >= self.max_disappeared or 
                track.consecutive_misses >= self.max_consecutive_misses):
                # Store as lost track for potential re-identification
                if track.embedding is not None:
                    self.lost_tracks[track.track_id] = {
                        'last_seen': self.frame_count,
                        'embedding': track.embedding.copy(),
                        'bbox': track.bbox.copy(),
                        'persistent_id': self.persistent_id_mapping.get(track.track_id, track.track_id)
                    }
                    self.logger.info(f"Track {track.track_id} lost, stored for re-identification")
        
        # Apply persistent re-identification
        self._apply_persistent_reid()
        
        # Save persistent data every 100 frames
        if self.frame_count % 100 == 0:
            self._save_persistent_data()
        
        self.tracks = [
            track for track in self.tracks
            if (track.time_since_update < self.max_disappeared and 
                track.consecutive_misses < self.max_consecutive_misses)
        ]
        
        tracked_faces = []
        for track in self.tracks:
            if track.state == "Confirmed":
                # Get persistent ID if available
                persistent_id = self.persistent_id_mapping.get(track.track_id, track.track_id)
                
                # Get or assign consistent display ID for UI
                display_id = self._get_or_assign_display_id(persistent_id)
                
                tracked_face = {
                    'track_id': track.track_id,
                    'persistent_id': persistent_id,  # Internal persistent ID
                    'display_id': display_id,  # Consistent display ID for UI
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'embedding': track.embedding,
                    'age': track.age,
                    'hits': track.hits,
                    'state': track.state
                }
                tracked_faces.append(tracked_face)
        
        return tracked_faces
    
    def _get_or_assign_display_id(self, persistent_id: int) -> int:
        """Get or assign a consistent display ID for UI purposes."""
        if persistent_id not in self.display_id_mapping:
            # Assign new display ID
            self.display_id_mapping[persistent_id] = self.next_display_id
            self.next_display_id += 1
            self.logger.info(f"Assigned display ID {self.display_id_mapping[persistent_id]} to persistent ID {persistent_id}")
        
        return self.display_id_mapping[persistent_id]
    
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
                    self.logger.debug(f"Track {track.track_id} vs Detection {j}: IoU={iou:.3f}, Embedding_cost={embedding_cost:.3f}")
                else:
                    self.logger.debug(f"Track {track.track_id} vs Detection {j}: IoU={iou:.3f}, No embedding comparison (track_emb: {track.embedding is not None}, det_emb: {embeddings[j] is not None})")
                
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
        """Calculate enhanced distance between embeddings using multiple metrics"""
        try:
            # Cosine similarity (primary metric)
            emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            cosine_similarity = np.dot(emb1_norm, emb2_norm)
            cosine_distance = 1.0 - cosine_similarity
            
            # Euclidean distance (normalized)
            euclidean_distance = np.linalg.norm(embedding1 - embedding2)
            euclidean_distance = euclidean_distance / (np.linalg.norm(embedding1) + np.linalg.norm(embedding2) + 1e-8)
            
            # Manhattan distance (normalized)
            manhattan_distance = np.sum(np.abs(embedding1 - embedding2))
            manhattan_distance = manhattan_distance / (np.sum(np.abs(embedding1)) + np.sum(np.abs(embedding2)) + 1e-8)
            
            # Combined distance (weighted average)
            combined_distance = (
                0.6 * cosine_distance +      # 60% weight on cosine similarity
                0.25 * euclidean_distance +  # 25% weight on euclidean distance
                0.15 * manhattan_distance    # 15% weight on manhattan distance
            )
            
            self.logger.debug(f"Embedding distances - Cosine: {cosine_distance:.3f}, Euclidean: {euclidean_distance:.3f}, Manhattan: {manhattan_distance:.3f}, Combined: {combined_distance:.3f}")
            
            return combined_distance
            
        except Exception as e:
            self.logger.error(f"Error calculating embedding distance: {e}")
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
        # Save current data before reset
        self._save_persistent_data()
        
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.persistent_tracks = {}
        self.lost_tracks = {}
        self.person_embeddings = {}
        self.next_persistent_id = 1
        self.persistent_id_mapping = {}
        self.display_id_mapping = {}
        self.next_display_id = 1
        self.logger.info("Tracking service reset")
    
    def _apply_persistent_reid(self):
        """Apply persistent person re-identification to maintain consistent track IDs."""
        current_track_ids = set(track.track_id for track in self.tracks)
        self.logger.info(f"Applying persistent ReID: {len(current_track_ids)} current tracks, {len(self.lost_tracks)} lost tracks")
        
        # Find tracks that are no longer active
        lost_track_ids = set(self.persistent_id_mapping.keys()) - current_track_ids
        
        for track_id in lost_track_ids:
            if track_id not in self.lost_tracks:
                # Store information about the lost track
                track = next((t for t in self.tracks if t.track_id == track_id), None)
                if track:
                    embedding = track.embedding
                    bbox = track.bbox
                    
                    self.lost_tracks[track_id] = {
                        'last_seen': self.frame_count,
                        'embedding': embedding,
                        'bbox': bbox,
                        'persistent_id': self.persistent_id_mapping[track_id]
                    }
        
        # Check new tracks for potential matches with lost tracks
        for track in self.tracks:
            if track.track_id not in self.persistent_id_mapping:
                self.logger.debug(f"Track {track.track_id} not in mapping, trying to find match")
                # Try to find a matching lost track
                persistent_id = self._find_matching_lost_track(track.embedding, track.bbox)
                
                if persistent_id is not None:
                    # Double-check that this persistent ID is not already assigned
                    if persistent_id not in self.persistent_id_mapping.values():
                        # Reuse existing persistent ID but keep unique track ID
                        old_track_id = track.track_id
                        self.persistent_id_mapping[track.track_id] = persistent_id
                        
                        # Remove the matched lost track to prevent duplicate matches
                        for lost_track_id, lost_data in list(self.lost_tracks.items()):
                            if lost_data.get('persistent_id') == persistent_id:
                                del self.lost_tracks[lost_track_id]
                                break
                        
                        self.logger.info(f"Re-identified person: Track {old_track_id} -> Persistent ID {persistent_id}")
                    else:
                        # Persistent ID is already assigned, assign new one
                        new_persistent_id = self.next_persistent_id
                        self.next_persistent_id += 1
                        self.persistent_id_mapping[track.track_id] = new_persistent_id
                        self.logger.warning(f"Persistent ID {persistent_id} already in use, assigned new ID {new_persistent_id} to Track {track.track_id}")
                else:
                    # Assign new persistent ID
                    persistent_id = self.next_persistent_id
                    self.next_persistent_id += 1
                    self.persistent_id_mapping[track.track_id] = persistent_id
                    self.logger.info(f"New person detected: Track {track.track_id} -> Persistent ID {persistent_id}")
            
            # Store embedding for future matching
            if track.embedding is not None:
                if track.track_id not in self.person_embeddings:
                    self.person_embeddings[track.track_id] = []
                self.person_embeddings[track.track_id].append(track.embedding)
                self.logger.info(f"Stored embedding for track {track.track_id}, total embeddings: {len(self.person_embeddings[track.track_id])}")
                
                # Keep only recent embeddings (last 10)
                if len(self.person_embeddings[track.track_id]) > 10:
                    self.person_embeddings[track.track_id] = self.person_embeddings[track.track_id][-10:]
        
        # Clean up old lost tracks
        tracks_to_remove = []
        for track_id, lost_data in self.lost_tracks.items():
            frames_since_lost = self.frame_count - lost_data['last_seen']
            if frames_since_lost > self.max_track_id_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.lost_tracks[track_id]
            if track_id in self.persistent_id_mapping:
                del self.persistent_id_mapping[track_id]
            if track_id in self.person_embeddings:
                del self.person_embeddings[track_id]
    
    def _find_matching_lost_track(self, embedding: Optional[np.ndarray]) -> Optional[int]:
        """Find a matching lost track based on embedding similarity."""
        self.logger.debug(f"_find_matching_lost_track called: embedding={embedding is not None}, lost_tracks={len(self.lost_tracks)}, person_embeddings={len(self.person_embeddings)}")
        if embedding is None:
            self.logger.debug("No embedding provided, returning None")
            return None
        
        best_match_id = None
        best_similarity = 0.0
        
        # Check if this persistent ID is already in use by ACTIVE tracks only
        used_persistent_ids = set()
        # Check active tracks
        for track in self.tracks:
            if track.track_id in self.persistent_id_mapping:
                used_persistent_ids.add(self.persistent_id_mapping[track.track_id])
        
        # First, check against existing person embeddings (for re-identification within same session)
        for track_id, embeddings_list in self.person_embeddings.items():
            if not embeddings_list:
                continue
            
            # Get the persistent ID for this track
            persistent_id = self.persistent_id_mapping.get(track_id)
            if persistent_id is None or persistent_id in used_persistent_ids:
                continue
            
            # Compare with all stored embeddings for this track
            for stored_embedding in embeddings_list:
                similarity = self._calculate_embedding_similarity(embedding, stored_embedding)
                
                if similarity > best_similarity and similarity > self.persistent_similarity_threshold:
                    best_similarity = similarity
                    best_match_id = persistent_id
                    self.logger.info(f"Found match in existing embeddings: track_id={track_id}, persistent_id={persistent_id}, similarity={similarity:.3f}")
                    # If we find a very good match, return immediately
                    if similarity > 0.7:  # Lowered threshold for immediate return
                        return best_match_id
                elif similarity > 0.2:  # Log potential matches even if below threshold
                    self.logger.debug(f"Potential match (below threshold): track_id={track_id}, persistent_id={persistent_id}, similarity={similarity:.3f}")
        
        # Then check against lost tracks
        for lost_track_id, lost_data in self.lost_tracks.items():
            lost_embedding = lost_data.get('embedding')
            if lost_embedding is None:
                continue
            
            persistent_id = lost_data.get('persistent_id')
            if persistent_id in used_persistent_ids:
                # This persistent ID is already in use by any track, skip
                continue
            
            # Calculate embedding similarity
            similarity = self._calculate_embedding_similarity(embedding, lost_embedding)
            
            # Use only embedding similarity for re-identification
            combined_score = similarity
            
            # Very permissive threshold for re-identification
            threshold = max(0.2, self.persistent_similarity_threshold)
            
            if combined_score > best_similarity and combined_score > threshold:
                best_similarity = combined_score
                best_match_id = persistent_id
                self.logger.info(f"Found potential match in lost tracks: persistent_id={persistent_id}, similarity={similarity:.3f}, combined={combined_score:.3f}")
                # If we find a very good match, return immediately
                if similarity > 0.7:  # Lowered threshold for immediate return
                    return best_match_id
            elif similarity > 0.2:  # Log potential matches even if below threshold
                self.logger.debug(f"Potential match in lost tracks (below threshold): persistent_id={persistent_id}, similarity={similarity:.3f}")
        
        if best_match_id is not None:
            self.logger.info(f"Re-identifying person with persistent ID {best_match_id} (similarity: {best_similarity:.3f})")
        
        return best_match_id
    
    def _find_matching_active_track(self, embedding: Optional[np.ndarray]) -> Optional[int]:
        """Find a matching active track based on embedding similarity (for same person with different head position)."""
        if embedding is None:
            return None
        
        best_match_id = None
        best_similarity = 0.0
        
        # Check against active tracks to see if this is the same person with different head position
        for track in self.tracks:
            if track.embedding is None:
                continue
            
            # Calculate similarity with active track
            similarity = self._calculate_embedding_similarity(embedding, track.embedding)
            
            # Use a higher threshold for active track matching (same person, different head position)
            if similarity > best_similarity and similarity > 0.6:  # Higher threshold for active tracks
                best_similarity = similarity
                best_match_id = self.persistent_id_mapping.get(track.track_id)
                self.logger.info(f"Found match with active track: track_id={track.track_id}, persistent_id={best_match_id}, similarity={similarity:.3f}")
        
        if best_match_id is not None:
            self.logger.info(f"Re-identifying with active track: persistent_id={best_match_id} (similarity: {best_similarity:.3f})")
        
        return best_match_id
    
    def _find_closest_existing_persistent_id(self, embedding: np.ndarray) -> Optional[int]:
        """Find the closest existing persistent ID based on embedding similarity."""
        if embedding is None:
            return None
        
        best_match_id = None
        best_similarity = 0.0
        
        # Get currently used persistent IDs by active tracks only
        used_persistent_ids = set()
        for track in self.tracks:
            if track.track_id in self.persistent_id_mapping:
                used_persistent_ids.add(self.persistent_id_mapping[track.track_id])
        
        # Check all existing person embeddings
        for track_id, embeddings_list in self.person_embeddings.items():
            if not embeddings_list:
                continue
            
            persistent_id = self.persistent_id_mapping.get(track_id)
            if persistent_id is None or persistent_id in used_persistent_ids:
                continue  # Skip if already in use by active track
            
            # Compare with all stored embeddings for this track
            for stored_embedding in embeddings_list:
                similarity = self._calculate_embedding_similarity(embedding, stored_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = persistent_id
        
        # Only return if similarity is above a reasonable threshold
        if best_similarity > 0.3:  # Lower threshold for closest match
            self.logger.info(f"Found closest existing persistent ID {best_match_id} with similarity {best_similarity:.3f}")
            return best_match_id
        
        return None
    
    def _calculate_embedding_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            self.logger.debug(f"Error calculating embedding similarity: {e}")
            return 0.0
    
    def _get_bbox_center(self, bbox: List[float]) -> List[float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    def _load_persistent_data(self):
        """Load persistent embeddings and person details from JSON files."""
        import os
        import json
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.embeddings_json_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.person_details_json_path), exist_ok=True)
        
        # Load embeddings
        if os.path.exists(self.embeddings_json_path):
            try:
                with open(self.embeddings_json_path, 'r') as f:
                    embeddings_data = json.load(f)
                    self.person_embeddings = {
                        int(k): [np.array(v) for v in embeddings_list] 
                        for k, embeddings_list in embeddings_data.items()
                    }
                    self.logger.info(f"Loaded {len(self.person_embeddings)} person embeddings from JSON")
            except Exception as e:
                self.logger.warning(f"Failed to load embeddings from JSON: {e}")
                self.person_embeddings = {}
        else:
            self.person_embeddings = {}
        
        # Load person details
        if os.path.exists(self.person_details_json_path):
            try:
                with open(self.person_details_json_path, 'r') as f:
                    person_details = json.load(f)
                    self.persistent_id_mapping = {int(k): v for k, v in person_details.get('persistent_id_mapping', {}).items()}
                    self.next_persistent_id = person_details.get('next_persistent_id', 1)
                    self.display_id_mapping = {int(k): v for k, v in person_details.get('display_id_mapping', {}).items()}
                    self.next_display_id = person_details.get('next_display_id', 1)
                    self.lost_tracks = {
                        int(k): {
                            'last_seen': v['last_seen'],
                            'embedding': np.array(v['embedding']) if v['embedding'] else None,
                            'bbox': v['bbox'],
                            'persistent_id': v['persistent_id']
                        }
                        for k, v in person_details.get('lost_tracks', {}).items()
                    }
                    self.logger.info(f"Loaded person details: {len(self.persistent_id_mapping)} mappings, next_id: {self.next_persistent_id}")
            except Exception as e:
                self.logger.warning(f"Failed to load person details from JSON: {e}")
                self.persistent_id_mapping = {}
                self.next_persistent_id = 1
                self.lost_tracks = {}
        else:
            self.persistent_id_mapping = {}
            self.next_persistent_id = 1
            self.lost_tracks = {}
    
    def _save_persistent_data(self):
        """Save persistent embeddings and person details to JSON files."""
        import os
        import json
        
        try:
            # Save embeddings
            embeddings_data = {
                str(k): [embedding.tolist() for embedding in embeddings_list]
                for k, embeddings_list in self.person_embeddings.items()
            }
            with open(self.embeddings_json_path, 'w') as f:
                json.dump(embeddings_data, f, indent=2)
            
            # Save person details
            person_details = {
                'persistent_id_mapping': {str(k): int(v) for k, v in self.persistent_id_mapping.items()},
                'next_persistent_id': int(self.next_persistent_id),
                'display_id_mapping': {str(k): int(v) for k, v in self.display_id_mapping.items()},
                'next_display_id': int(self.next_display_id),
                'lost_tracks': {
                    str(k): {
                        'last_seen': int(v['last_seen']),
                        'embedding': v['embedding'].tolist() if v['embedding'] is not None else None,
                        'bbox': [float(x) for x in v['bbox']],
                        'persistent_id': int(v['persistent_id'])
                    }
                    for k, v in self.lost_tracks.items()
                }
            }
            with open(self.person_details_json_path, 'w') as f:
                json.dump(person_details, f, indent=2)
            
            self.logger.info(f"Saved persistent data: {len(self.person_embeddings)} embeddings, {len(self.persistent_id_mapping)} mappings")
            
        except Exception as e:
            self.logger.error(f"Failed to save persistent data to JSON: {e}")
    
    def save_persistent_data(self):
        """Public method to save persistent data."""
        self._save_persistent_data()
    
    def _extract_person_embedding(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Extract an enhanced person embedding from the bounding box region.
        This is an improved feature extraction for person re-identification.
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Extract person region
            person_region = frame[y1:y2, x1:x2]
            
            if person_region.size == 0:
                self.logger.debug("Empty person region, skipping embedding extraction")
                return None
            
            # Resize to standard size for consistent features
            person_region = cv2.resize(person_region, (64, 128))
            
            # Convert to grayscale for simpler features
            gray = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
            
            # Extract histogram features (simple but effective for person re-id)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist.flatten()
            
            # Normalize histogram
            hist = hist / (np.sum(hist) + 1e-8)
            
            # Add spatial features (color distribution in different regions)
            h, w = gray.shape
            top_region = gray[:h//3, :]
            mid_region = gray[h//3:2*h//3, :]
            bot_region = gray[2*h//3:, :]
            
            top_hist = cv2.calcHist([top_region], [0], None, [16], [0, 256]).flatten()
            mid_hist = cv2.calcHist([mid_region], [0], None, [16], [0, 256]).flatten()
            bot_hist = cv2.calcHist([bot_region], [0], None, [16], [0, 256]).flatten()
            
            # Add color features from original image
            color_region = person_region
            color_region = cv2.resize(color_region, (32, 64))  # Smaller for efficiency
            
            # Extract color histograms for each channel
            b_hist = cv2.calcHist([color_region], [0], None, [16], [0, 256]).flatten()
            g_hist = cv2.calcHist([color_region], [1], None, [16], [0, 256]).flatten()
            r_hist = cv2.calcHist([color_region], [2], None, [16], [0, 256]).flatten()
            
            # Normalize color histograms
            b_hist = b_hist / (np.sum(b_hist) + 1e-8)
            g_hist = g_hist / (np.sum(g_hist) + 1e-8)
            r_hist = r_hist / (np.sum(r_hist) + 1e-8)
            
            # Add texture features using LBP-like approach
            texture_features = self._extract_texture_features(gray)
            
            # Combine all features
            features = np.concatenate([
                hist,           # 32 grayscale histogram
                top_hist,       # 16 top region
                mid_hist,       # 16 mid region  
                bot_hist,       # 16 bottom region
                b_hist,         # 16 blue channel
                g_hist,         # 16 green channel
                r_hist,         # 16 red channel
                texture_features # 8 texture features
            ])
            
            # Normalize the final feature vector
            features = features / (np.linalg.norm(features) + 1e-8)
            
            self.logger.debug(f"Extracted person embedding with {len(features)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting person embedding: {e}")
            return None
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract simple texture features using gradient information."""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Extract simple texture statistics
            mean_magnitude = np.mean(magnitude)
            std_magnitude = np.std(magnitude)
            max_magnitude = np.max(magnitude)
            min_magnitude = np.min(magnitude)
            
            # Calculate gradient direction statistics
            direction = np.arctan2(grad_y, grad_x)
            mean_direction = np.mean(direction)
            std_direction = np.std(direction)
            
            # Add some spatial texture features
            h, w = gray_image.shape
            center_region = gray_image[h//4:3*h//4, w//4:3*w//4]
            center_texture = np.std(center_region)
            
            # Edge density
            edges = cv2.Canny(gray_image, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            
            texture_features = np.array([
                mean_magnitude, std_magnitude, max_magnitude, min_magnitude,
                mean_direction, std_direction, center_texture, edge_density
            ])
            
            return texture_features
            
        except Exception as e:
            self.logger.error(f"Error extracting texture features: {e}")
            return np.zeros(8)
    
    def _extract_embeddings_parallel(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
        """Extract embeddings for multiple detections in parallel."""
        try:
            # Prepare data for parallel processing
            bboxes = [detection['bbox'] for detection in detections]
            
            # Use multiprocessing to extract embeddings
            with Pool(processes=self.max_workers) as pool:
                extract_func = partial(_extract_person_embedding_worker, frame)
                embeddings = pool.map(extract_func, bboxes)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error in parallel embedding extraction: {e}")
            # Fallback to sequential processing
            embeddings = []
            for detection in detections:
                embedding = self._extract_person_embedding(frame, detection['bbox'])
                embeddings.append(embedding)
            return embeddings


def _extract_person_embedding_worker(frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
    """
    Worker function for parallel embedding extraction.
    This function needs to be at module level for multiprocessing.
    Uses the same enhanced embedding extraction as the main function.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Extract person region
        person_region = frame[y1:y2, x1:x2]
        
        if person_region.size == 0:
            return None
        
        # Resize to standard size for consistent features
        person_region = cv2.resize(person_region, (64, 128))
        
        # Convert to grayscale for simpler features
        gray = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
        
        # Extract histogram features (simple but effective for person re-id)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = hist.flatten()
        
        # Normalize histogram
        hist = hist / (np.sum(hist) + 1e-8)
        
        # Add spatial features (color distribution in different regions)
        h, w = gray.shape
        top_region = gray[:h//3, :]
        mid_region = gray[h//3:2*h//3, :]
        bot_region = gray[2*h//3:, :]
        
        top_hist = cv2.calcHist([top_region], [0], None, [16], [0, 256]).flatten()
        mid_hist = cv2.calcHist([mid_region], [0], None, [16], [0, 256]).flatten()
        bot_hist = cv2.calcHist([bot_region], [0], None, [16], [0, 256]).flatten()
        
        # Add color features from original image
        color_region = person_region
        color_region = cv2.resize(color_region, (32, 64))  # Smaller for efficiency
        
        # Extract color histograms for each channel
        b_hist = cv2.calcHist([color_region], [0], None, [16], [0, 256]).flatten()
        g_hist = cv2.calcHist([color_region], [1], None, [16], [0, 256]).flatten()
        r_hist = cv2.calcHist([color_region], [2], None, [16], [0, 256]).flatten()
        
        # Normalize color histograms
        b_hist = b_hist / (np.sum(b_hist) + 1e-8)
        g_hist = g_hist / (np.sum(g_hist) + 1e-8)
        r_hist = r_hist / (np.sum(r_hist) + 1e-8)
        
        # Add texture features using LBP-like approach
        texture_features = _extract_texture_features_worker(gray)
        
        # Combine all features
        features = np.concatenate([
            hist,           # 32 grayscale histogram
            top_hist,       # 16 top region
            mid_hist,       # 16 mid region  
            bot_hist,       # 16 bottom region
            b_hist,         # 16 blue channel
            g_hist,         # 16 green channel
            r_hist,         # 16 red channel
            texture_features # 8 texture features
        ])
        
        # Normalize the final feature vector
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
        
    except Exception as e:
        return None

def _extract_texture_features_worker(gray_image: np.ndarray) -> np.ndarray:
    """Extract simple texture features using gradient information (worker version)."""
    try:
        # Calculate gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Extract simple texture statistics
        mean_magnitude = np.mean(magnitude)
        std_magnitude = np.std(magnitude)
        max_magnitude = np.max(magnitude)
        min_magnitude = np.min(magnitude)
        
        # Calculate gradient direction statistics
        direction = np.arctan2(grad_y, grad_x)
        mean_direction = np.mean(direction)
        std_direction = np.std(direction)
        
        # Add some spatial texture features
        h, w = gray_image.shape
        center_region = gray_image[h//4:3*h//4, w//4:3*w//4]
        center_texture = np.std(center_region)
        
        # Edge density
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        texture_features = np.array([
            mean_magnitude, std_magnitude, max_magnitude, min_magnitude,
            mean_direction, std_direction, center_texture, edge_density
        ])
        
        return texture_features
        
    except Exception as e:
        return np.zeros(8)
