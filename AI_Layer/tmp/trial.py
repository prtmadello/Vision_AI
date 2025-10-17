"""
Multi-Camera Person Tracking and Action Recognition System
Self-contained implementation with YOLO detection, ReID embeddings, and action recognition
"""

import cv2
import numpy as np
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from collections import deque, defaultdict
from datetime import datetime, timedelta
import logging

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install ultralytics package.")

# Try to import InsightFace for embeddings
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace not available. Install insightface package.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GlobalPersonRecord:
    """Record for a globally tracked person across all cameras"""
    
    def __init__(self, global_id: int, embedding: np.ndarray, camera_id: int, 
                 bbox: List[float], timestamp: float):
        self.global_id = global_id
        self.embedding = embedding
        self.last_seen_time = timestamp
        self.last_camera_id = camera_id
        self.last_bbox = bbox
        self.track_history = []  # List of (camera_id, bbox, timestamp)
        self.action_history = []  # List of (action, timestamp)
        self.embedding_history = deque(maxlen=10)  # Keep last 10 embeddings for smoothing
        self.embedding_history.append(embedding)
        
    def update(self, embedding: np.ndarray, camera_id: int, bbox: List[float], 
               timestamp: float, action: str = None):
        """Update person record with new detection"""
        self.embedding = self._update_embedding(embedding)
        self.last_seen_time = timestamp
        self.last_camera_id = camera_id
        self.last_bbox = bbox
        
        # Add to history
        self.track_history.append((camera_id, bbox, timestamp))
        if action:
            self.action_history.append((action, timestamp))
            
        # Keep history limited
        if len(self.track_history) > 100:
            self.track_history = self.track_history[-50:]
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-25:]
    
    def _update_embedding(self, new_embedding: np.ndarray) -> np.ndarray:
        """Update embedding using moving average"""
        self.embedding_history.append(new_embedding)
        
        # Use exponential moving average for smoother embeddings
        alpha = 0.3  # Learning rate
        if len(self.embedding_history) == 1:
            return new_embedding
        
        # Weighted average with more weight on recent embeddings
        weights = np.exp(np.linspace(-1, 0, len(self.embedding_history)))
        weights = weights / weights.sum()
        
        weighted_embedding = np.zeros_like(new_embedding)
        for i, emb in enumerate(self.embedding_history):
            weighted_embedding += weights[i] * emb
            
        return weighted_embedding
    
    def is_active(self, current_time: float, timeout: float = 30.0) -> bool:
        """Check if person is still active based on last seen time"""
        return (current_time - self.last_seen_time) < timeout


class ActionBuffer:
    """Buffer for action recognition with sliding window"""
    
    def __init__(self, window_size: int = 16, max_buffers: int = 10):
        self.window_size = window_size
        self.max_buffers = max_buffers
        self.buffers = {}  # global_id -> deque of frames
        
    def add_frame(self, global_id: int, frame: np.ndarray, bbox: List[float]):
        """Add frame to buffer for specific global_id"""
        if global_id not in self.buffers:
            self.buffers[global_id] = deque(maxlen=self.window_size)
        
        # Crop person from frame
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            # Resize to standard size for action recognition
            crop = cv2.resize(crop, (224, 224))
            self.buffers[global_id].append(crop)
    
    def get_action_sequence(self, global_id: int) -> Optional[np.ndarray]:
        """Get action sequence if buffer is full"""
        if global_id not in self.buffers:
            return None
        
        buffer = self.buffers[global_id]
        if len(buffer) < self.window_size:
            return None
        
        # Convert to numpy array for action model
        sequence = np.array(list(buffer))
        return sequence
    
    def cleanup_inactive(self, active_global_ids: set):
        """Remove buffers for inactive global IDs"""
        inactive_ids = set(self.buffers.keys()) - active_global_ids
        for global_id in inactive_ids:
            del self.buffers[global_id]


class SimpleTracker:
    """Simple tracker for per-camera tracking"""
    
    def __init__(self):
        self.tracks = {}  # track_id -> track_info
        self.next_id = 1
        self.max_distance = 100  # Maximum distance for track association
        
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update tracks with new detections"""
        tracks = []
        
        # Simple IoU-based tracking
        for detection in detections:
            bbox = detection['bbox']
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            
            # Find closest existing track
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, track_info in self.tracks.items():
                track_center = track_info['center']
                distance = np.sqrt((center[0] - track_center[0])**2 + (center[1] - track_center[1])**2)
                
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id]['bbox'] = bbox
                self.tracks[best_track_id]['center'] = center
                self.tracks[best_track_id]['confidence'] = detection['confidence']
                track_id = best_track_id
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'bbox': bbox,
                    'center': center,
                    'confidence': detection['confidence']
                }
            
            tracks.append({
                'track_id': track_id,
                'bbox': bbox,
                'confidence': detection['confidence']
            })
        
        # Remove old tracks (simple timeout)
        current_time = time.time()
        tracks_to_remove = []
        for track_id, track_info in self.tracks.items():
            if current_time - track_info.get('last_seen', current_time) > 2.0:  # 2 second timeout
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return tracks


class SimpleActionRecognizer:
    """Simple action recognition based on motion patterns"""
    
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        
    def predict_action(self, frame_sequence: np.ndarray) -> str:
        """Predict action from sequence of frames"""
        if len(frame_sequence) < 8:
            return "unknown"
        
        # Simple motion-based action recognition
        motion_scores = []
        
        for i in range(1, len(frame_sequence)):
            # Convert to grayscale
            prev_gray = cv2.cvtColor(frame_sequence[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame_sequence[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, 
                np.array([[112, 112]], dtype=np.float32).reshape(-1, 1, 2),
                None
            )[0]
            
            if flow is not None and len(flow) > 0:
                motion_magnitude = np.linalg.norm(flow[0])
                motion_scores.append(motion_magnitude)
        
        if not motion_scores:
            return "standing"
        
        avg_motion = np.mean(motion_scores)
        
        # Simple thresholds for action classification
        if avg_motion < 2.0:
            return "standing"
        elif avg_motion < 5.0:
            return "walking"
        elif avg_motion < 10.0:
            return "running"
        else:
            return "fast_movement"


class MultiCameraTracker:
    """Multi-camera person tracking with global ID management"""
    
    def __init__(self):
        self.logger = logger
        
        # Global tracking state
        self.global_id_counter = 0
        self.global_db = {}  # global_id -> GlobalPersonRecord
        self.action_buffer = ActionBuffer(window_size=16)
        self.action_recognizer = SimpleActionRecognizer()
        
        # Configuration
        self.similarity_threshold = 0.6  # Cosine similarity threshold
        self.max_time_gap = 5.0  # Maximum time gap for matching (seconds)
        self.inactive_timeout = 30.0  # Timeout for inactive persons (seconds)
        
        # Initialize models
        self._initialize_models()
        
        # Camera management
        self.cameras = {}  # camera_id -> camera_info
        self.trackers = {}  # camera_id -> simple tracker
        
        self.logger.info("Multi-camera tracker initialized")
    
    def _initialize_models(self):
        """Initialize AI models directly"""
        try:
            # Initialize YOLO for person detection
            if YOLO_AVAILABLE:
                model_path = "/home/prithiviraj/Vision_AI/AI_Layer/models/yolov8n.pt"
                if Path(model_path).exists():
                    self.yolo_model = YOLO(model_path)
                    self.logger.info(f"YOLO model loaded: {model_path}")
                else:
                    self.yolo_model = YOLO('yolov8n.pt')  # Download if not found
                    self.logger.info("YOLO model downloaded and loaded")
            else:
                self.yolo_model = None
                self.logger.warning("YOLO not available")
            
            # Initialize InsightFace for embeddings
            if INSIGHTFACE_AVAILABLE:
                self.face_analyzer = FaceAnalysis(name='buffalo_l')
                self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
                self.logger.info("InsightFace initialized")
            else:
                self.face_analyzer = None
                self.logger.warning("InsightFace not available")
                
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def add_camera(self, camera_id: int, video_source: str):
        """Add a camera to the tracking system"""
        self.cameras[camera_id] = {
            'source': video_source,
            'cap': None,
            'frame_count': 0,
            'fps': 30,
            'last_frame_time': 0
        }
        
        # Initialize camera capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {camera_id}: {video_source}")
            return False
        
        self.cameras[camera_id]['cap'] = cap
        self.cameras[camera_id]['fps'] = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Initialize simple tracker for this camera
        self.trackers[camera_id] = SimpleTracker()
        
        self.logger.info(f"Camera {camera_id} added: {video_source}")
        return True
    
    def _detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons in frame using YOLO"""
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(frame, conf=0.5, iou=0.45, max_det=100)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Only detect persons (class 0 in COCO)
                        if class_id == 0:
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': 'person'
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in person detection: {e}")
            return []
    
    def _extract_person_embedding(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Extract person embedding from frame and bounding box"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return None
            
            # Try to use InsightFace for face-based embedding
            if self.face_analyzer is not None:
                try:
                    faces = self.face_analyzer.get(crop)
                    if faces:
                        # Use the largest face
                        largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                        embedding = largest_face.embedding
                        return embedding
                except Exception as e:
                    self.logger.debug(f"InsightFace failed: {e}")
            
            # Fallback: use a simple feature extraction
            # Resize and normalize
            crop_resized = cv2.resize(crop, (224, 224))
            crop_normalized = crop_resized.astype(np.float32) / 255.0
            
            # Simple feature extraction using color histograms and texture
            # Color histogram
            hist_b = cv2.calcHist([crop_normalized], [0], None, [32], [0, 1])
            hist_g = cv2.calcHist([crop_normalized], [1], None, [32], [0, 1])
            hist_r = cv2.calcHist([crop_normalized], [2], None, [32], [0, 1])
            
            # Texture features using LBP-like approach
            gray = cv2.cvtColor(crop_normalized, cv2.COLOR_BGR2GRAY)
            texture_features = []
            for i in range(0, gray.shape[0]-8, 8):
                for j in range(0, gray.shape[1]-8, 8):
                    patch = gray[i:i+8, j:j+8]
                    if patch.size > 0:
                        texture_features.append(np.mean(patch))
            
            # Combine features
            color_features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
            texture_features = np.array(texture_features[:100])  # Limit to 100 features
            
            if len(texture_features) < 100:
                texture_features = np.pad(texture_features, (0, 100 - len(texture_features)), 'constant')
            
            features = np.concatenate([color_features, texture_features])
            features = features / (np.linalg.norm(features) + 1e-8)  # L2 normalize
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting person embedding: {e}")
            return None
    
    def _find_best_match(self, embedding: np.ndarray, camera_id: int, current_time: float) -> Optional[int]:
        """Find best matching global ID for the given embedding"""
        best_score = self.similarity_threshold
        matched_global_id = None
        
        for global_id, record in self.global_db.items():
            # Check time gap constraint
            time_gap = current_time - record.last_seen_time
            if time_gap > self.max_time_gap:
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(embedding, record.embedding)
            
            if similarity > best_score:
                best_score = similarity
                matched_global_id = global_id
        
        return matched_global_id
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Ensure embeddings are normalized
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # Cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _process_camera_frame(self, camera_id: int, frame: np.ndarray, current_time: float) -> List[Dict[str, Any]]:
        """Process a single frame from a camera"""
        results = []
        
        try:
            # Step 1: Person Detection
            detections = self._detect_persons(frame)
            
            if not detections:
                return results
            
            # Step 2: Per-camera tracking
            if camera_id in self.trackers:
                tracks = self.trackers[camera_id].update(detections)
            else:
                # Fallback to simple tracking
                tracks = []
                for i, det in enumerate(detections):
                    tracks.append({
                        'track_id': i,
                        'bbox': det['bbox'],
                        'confidence': det['confidence']
                    })
            
            # Step 3: Process each track
            for track in tracks:
                bbox = track['bbox']
                
                # Extract embedding
                embedding = self._extract_person_embedding(frame, bbox)
                if embedding is None:
                    continue
                
                # Step 4: Cross-camera matching / Global ID assignment
                matched_global_id = self._find_best_match(embedding, camera_id, current_time)
                
                if matched_global_id is None:
                    # Create new global person
                    self.global_id_counter += 1
                    matched_global_id = self.global_id_counter
                    
                    self.global_db[matched_global_id] = GlobalPersonRecord(
                        matched_global_id, embedding, camera_id, bbox, current_time
                    )
                    
                    self.logger.info(f"New global person created: ID {matched_global_id}")
                else:
                    # Update existing global record
                    self.global_db[matched_global_id].update(
                        embedding, camera_id, bbox, current_time
                    )
                
                # Step 5: Action Detection
                self.action_buffer.add_frame(matched_global_id, frame, bbox)
                action_sequence = self.action_buffer.get_action_sequence(matched_global_id)
                
                action = "unknown"
                if action_sequence is not None:
                    action = self.action_recognizer.predict_action(action_sequence)
                    if action != "unknown":
                        self.global_db[matched_global_id].update(
                            embedding, camera_id, bbox, current_time, action
                        )
                
                # Store result
                result = {
                    'global_id': matched_global_id,
                    'camera_id': camera_id,
                    'bbox': bbox,
                    'timestamp': current_time,
                    'action': action,
                    'confidence': track.get('confidence', 0.0)
                }
                results.append(result)
                
        except Exception as e:
            self.logger.error(f"Error processing camera {camera_id} frame: {e}")
        
        return results
    
    def _cleanup_inactive_persons(self, current_time: float):
        """Remove inactive persons from global database"""
        inactive_ids = []
        
        for global_id, record in self.global_db.items():
            if not record.is_active(current_time, self.inactive_timeout):
                inactive_ids.append(global_id)
        
        for global_id in inactive_ids:
            del self.global_db[global_id]
            self.logger.info(f"Removed inactive person: ID {global_id}")
        
        # Clean up action buffers
        active_ids = set(self.global_db.keys())
        self.action_buffer.cleanup_inactive(active_ids)
    
    def process_frame(self, current_time: float = None) -> List[Dict[str, Any]]:
        """Process one frame from all cameras"""
        if current_time is None:
            current_time = time.time()
        
        all_results = []
        
        # Process each camera
        for camera_id, camera_info in self.cameras.items():
            cap = camera_info['cap']
            if cap is None or not cap.isOpened():
                continue
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process frame
            results = self._process_camera_frame(camera_id, frame, current_time)
            all_results.extend(results)
        
        # Cleanup inactive persons
        self._cleanup_inactive_persons(current_time)
        
        return all_results
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get statistics about global tracking"""
        current_time = time.time()
        active_persons = sum(1 for record in self.global_db.values() 
                           if record.is_active(current_time, self.inactive_timeout))
        
        return {
            'total_global_ids': self.global_id_counter,
            'active_persons': active_persons,
            'total_tracked': len(self.global_db),
            'cameras_active': len([c for c in self.cameras.values() if c['cap'] and c['cap'].isOpened()]),
            'action_buffers': len(self.action_buffer.buffers)
        }
    
    def get_person_history(self, global_id: int) -> Optional[Dict[str, Any]]:
        """Get history for a specific global person ID"""
        if global_id not in self.global_db:
            return None
        
        record = self.global_db[global_id]
        return {
            'global_id': global_id,
            'last_seen': record.last_seen_time,
            'last_camera': record.last_camera_id,
            'track_history': record.track_history,
            'action_history': record.action_history,
            'is_active': record.is_active(time.time(), self.inactive_timeout)
        }
    
    def cleanup(self):
        """Clean up resources"""
        for camera_info in self.cameras.values():
            if camera_info['cap']:
                camera_info['cap'].release()
        
        cv2.destroyAllWindows()
        self.logger.info("Multi-camera tracker cleaned up")


def main():
    """Main function to demonstrate multi-camera tracking"""
    logger.info("Starting Multi-Camera Person Tracking Demo")
    
    # Initialize tracker
    tracker = MultiCameraTracker()
    
    # Add sample videos as cameras
    sample_videos = [
        "/home/prithiviraj/Vision_AI/AI_Layer/input/videos/sam.mp4",
    ]
    
    for i, video_path in enumerate(sample_videos):
        if Path(video_path).exists():
            tracker.add_camera(i + 1, video_path)
        else:
            logger.warning(f"Video not found: {video_path}")
    
    if not tracker.cameras:
        logger.error("No cameras available. Exiting.")
        return
    
    logger.info(f"Processing {len(tracker.cameras)} cameras...")
    
    # Process frames
    frame_count = 0
    max_frames = 500  # Limit for demo
    
    try:
        while frame_count < max_frames:
            current_time = time.time()
            
            # Process all cameras
            results = tracker.process_frame(current_time)
            
            # Log results
            if results:
                logger.info(f"Frame {frame_count}: {len(results)} detections")
                for result in results:
                    logger.info(f"  Global ID {result['global_id']} in camera {result['camera_id']} "
                              f"action: {result['action']} bbox: {result['bbox']}")
            
            # Print statistics every 50 frames
            if frame_count % 50 == 0:
                stats = tracker.get_global_statistics()
                logger.info(f"Statistics: {stats}")
            
            frame_count += 1
            
            # Small delay to prevent overwhelming
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        tracker.cleanup()
        logger.info("Demo completed")


if __name__ == "__main__":
    main()
