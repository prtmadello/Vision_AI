"""
Video processing service with face detection, tracking, and ReID
"""

import cv2
import numpy as np
import csv
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from utils.logger import setup_logger
from utils.image_processor import ImageProcessor
from services.strongsort_tracking_service import StrongSORTTrackingService

logger = setup_logger(__name__)


class VideoProcessingService:
    """Video processing service with face detection, tracking, and ReID"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        
        # Video processing settings
        self.frame_skip = config.get('frame_skip', 1)
        self.max_frames = config.get('max_frames', None)
        self.video_fps = config.get('fps', 30)
        
        # Initialize image processor for drawing
        self.image_processor = ImageProcessor()
        self.video_codec = config.get('video_codec', 'mp4v')
        self.show_progress = config.get('show_progress', True)
        
        # Output settings
        self.save_annotated_video = config.get('save_annotated_video', True)
        
        # Multiprocessing settings
        self.use_multiprocessing = config.get('multiprocessing', {}).get('enabled', True)
        self.max_workers = config.get('multiprocessing', {}).get('max_workers', min(4, cpu_count()))
        self.logger.info(f"Video processing multiprocessing enabled: {self.use_multiprocessing} (workers: {self.max_workers})")
        
        # Initialize StrongSORT if enabled
        strongsort_config = config.get('strongsort_tracking', {})
        if strongsort_config.get('enabled', False):
            self.logger.info(f"Initializing StrongSORT with config: {strongsort_config}")
            self.strongsort_service = StrongSORTTrackingService(strongsort_config)
            self.logger.info("StrongSORT service initialized successfully")
        else:
            self.logger.info("StrongSORT is disabled in configuration")
            self.strongsort_service = None
        # Data output configuration
        data_output_config = config.get('data_output', {})
        self.data_output_mode = data_output_config.get('mode', 'csv')  # 'csv' or 'kafka'
        
        # CSV settings - only enable if mode is 'csv' or not specified
        self.save_csv = config.get('save_csv', True) and self.data_output_mode == 'csv'
        self.output_path = Path(config.get('output_path', 'output'))
        # Optional global face stream CSV path for API layer consumption
        self.global_stream_csv_path = config.get('data_output_csv_path', None) if self.data_output_mode == 'csv' else None
        
        # Log data output mode
        self.logger.info(f"Video processing data output mode: {self.data_output_mode}")
        if self.data_output_mode == 'kafka':
            self.logger.info("CSV writing disabled - using Kafka for data output")
        else:
            self.logger.info("CSV writing enabled - using CSV for data output")
        
        # Track name persistence to avoid pulsing
        face_recognition_config = config.get('face_recognition', {})
        name_persistence_config = face_recognition_config.get('name_persistence', {})
        
        self.track_name_cache = {}  # track_id -> {'name': str, 'confidence': float, 'frames_seen': int, 'last_seen': int}
        self.name_persistence_enabled = name_persistence_config.get('enabled', True)
        self.name_confidence_threshold = name_persistence_config.get('confidence_threshold', 0.7)
        self.name_persistence_frames = name_persistence_config.get('persistence_frames', 10)
        self.name_improvement_threshold = name_persistence_config.get('improvement_threshold', 0.1)
        
        # Box colors configuration
        self.box_colors = {
            'face_known': (0, 255, 0),      # Green
            'face_unknown': (0, 0, 255),    # Red
            'human_known': (255, 0, 0),     # Blue
            'human_unknown': (0, 255, 255), # Yellow
            'track_confirmed': (0, 255, 0),  # Green
            'track_tentative': (0, 255, 255) # Yellow
        }
    
    def process_video_with_face_detection(
        self,
        video_path: str,
        detection_service,
        face_processing_service,
        vectorization_service,
        tracking_service,
        database_service=None,
        reference_embeddings=None,
        kafka_service=None
    ) -> Dict[str, Any]:
        """Process video for face detection and recognition"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'success': False, 'error': 'Could not open video file'}
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or self.video_fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize tracking
            tracking_service.reset()
            
            # Prepare output
            output_video_path = None
            if self.save_annotated_video:
                output_video_path, output_video_file_path = self._setup_output_video(video_path, width, height, fps)
            
            # Process frames
            all_detections = []
            frame_count = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Detect faces
                face_detections = detection_service.detect_faces(frame)
                print(f"DEBUG: Face detections returned: {len(face_detections)}")
                self.logger.info(f"Face detections returned: {len(face_detections)}")
                
                # Process faces
                processed_faces = face_processing_service.process_face_detections(
                    frame, face_detections
                )
                
                # Apply tracking
                tracked_faces = tracking_service.update(face_detections, frame)
                
                # Recognize faces if reference embeddings available
                recognitions = []
                if reference_embeddings:
                    for i, face_info in enumerate(processed_faces):
                        face_crop = face_info['face_crop']
                        # Use the improved vectorization method for pre-cropped faces
                        embedding = vectorization_service.extract_embedding_from_crop(face_crop)
                        if embedding is None:
                            # Fallback: try general extraction on the crop
                            embedding = vectorization_service.extract_embedding(face_crop)
                        
                        if embedding is not None:
                            match_result = vectorization_service.find_best_match(
                                embedding, reference_embeddings
                            )
                            
                            recognition = {
                                'face_id': i,
                                'bbox': face_info['original_bbox'],
                                'confidence': face_info['confidence'],
                                'person_id': match_result['person_id'],
                                'person_name': match_result.get('person_name', 'Unknown'),
                                'match_confidence': match_result['confidence'],
                                'is_known': match_result['is_known'],
                                'status': match_result.get('status') if match_result.get('is_known') else None,
                                'track_id': tracked_faces[i].get('track_id', 'N/A') if i < len(tracked_faces) else 'N/A'
                            }
                            recognitions.append(recognition)
                
                # Store tracking data in database
                if database_service:
                    for i, detection in enumerate(face_detections):
                        track_id = tracked_faces[i].get('track_id', 0) if i < len(tracked_faces) else 0
                        person_id = recognitions[i].get('person_id', 'Unknown') if i < len(recognitions) else 'Unknown'
                        
                        database_service.store_tracking_data(
                            track_id=track_id,
                            person_id=person_id,
                            frame_id=frame_count,
                            bbox=detection['bbox'],
                            confidence=detection['confidence'],
                            track_age=tracked_faces[i].get('age', 0) if i < len(tracked_faces) else 0,
                            track_hits=tracked_faces[i].get('hits', 0) if i < len(tracked_faces) else 0,
                            track_state=tracked_faces[i].get('state', 'Unknown') if i < len(tracked_faces) else 'Unknown',
                            timestamp=frame_count / fps,
                            source=video_path
                        )
                
                # Annotate frame
                if self.save_annotated_video:
                    annotated_frame = self._annotate_frame_with_detections(
                        frame, face_detections, recognitions, tracked_faces
                    )
                    output_video_path.write(annotated_frame)
                
                # Store detection data
                frame_detections = []
                if face_detections:
                    self.logger.debug(f"Processing {len(face_detections)} face detections")
                    for i, detection in enumerate(face_detections):
                        # Convert numpy types to Python native types for JSON serialization
                        track_id = tracked_faces[i].get('track_id', 'N/A') if i < len(tracked_faces) else 'N/A'
                        if isinstance(track_id, np.integer):
                            track_id = int(track_id)
                        
                        detection_data = {
                            'frame_id': int(frame_count),
                            'detection_id': int(i),
                            'track_id': track_id,
                            'person_id': recognitions[i].get('person_id', 'Unknown') if i < len(recognitions) else 'Unknown',
                            'person_name': recognitions[i].get('person_name', 'Unknown') if i < len(recognitions) else 'Unknown',
                            'match_confidence': float(recognitions[i].get('match_confidence', 0.0)) if i < len(recognitions) else 0.0,
                            'status': recognitions[i].get('status', 'unknown') if i < len(recognitions) else 'unknown',
                            'bbox_x1': float(detection['bbox'][0]),
                            'bbox_y1': float(detection['bbox'][1]),
                            'bbox_x2': float(detection['bbox'][2]),
                            'bbox_y2': float(detection['bbox'][3]),
                            'confidence': float(detection['confidence']),
                            'track_age': int(tracked_faces[i].get('age', 0)) if i < len(tracked_faces) else 0,
                            'track_hits': int(tracked_faces[i].get('hits', 0)) if i < len(tracked_faces) else 0,
                            'track_state': tracked_faces[i].get('state', 'Unknown') if i < len(tracked_faces) else 'Unknown',
                            'timestamp': float(frame_count / fps),
                            'source': str(video_path)
                        }
                        frame_detections.append(detection_data)
                else:
                    self.logger.debug("No face detections in this frame - face detection may have failed")
                
                all_detections.extend(frame_detections)

                # Stream to Kafka if available
                if kafka_service:
                    for d in frame_detections:
                        kafka_service.send_detection(d)
                    if frame_detections:
                        self.logger.info(f"Sent {len(frame_detections)} face detections to Kafka")
                    else:
                        self.logger.debug("No face detections to send to Kafka")
                else:
                    self.logger.warning("Kafka service not available - face detection data not being streamed")
                processed_frames += 1
                frame_count += 1
                
                # Show progress
                if self.show_progress and frame_count % 30 == 0:
                    self.logger.info(f"Processed {frame_count}/{total_frames} frames")
                
                # Limit frames if specified
                if self.max_frames and processed_frames >= self.max_frames:
                    break
            
            cap.release()
            if output_video_path:
                output_video_path.release()
            
            # Save CSV if requested
            csv_path = None
            if self.save_csv and all_detections:
                csv_path = self._save_detections_csv(all_detections, video_path)
                # Also append to global face stream CSV for real-time consumers
                if self.global_stream_csv_path:
                    try:
                        self._append_to_global_stream_csv(all_detections, self.global_stream_csv_path)
                    except Exception as e:
                        logger.error(f"Failed to append to global stream CSV: {e}")
            
            return {
                'success': True,
                'total_frames': frame_count,
                'processed_frames': processed_frames,
                'total_detections': len(all_detections),
                'output_video': str(output_video_file_path) if self.save_annotated_video else None,
                'csv_file': csv_path
            }
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_video_with_human_detection(
        self,
        video_path: str,
        detection_service,
        tracking_service,
        database_service=None,
        kafka_service=None,
        face_recognition_service=None
    ) -> Dict[str, Any]:
        """Process video for human detection and ReID tracking"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'success': False, 'error': 'Could not open video file'}
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or self.video_fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize tracking
            tracking_service.reset()
            
            # Prepare output
            output_video_path = None
            if self.save_annotated_video:
                output_video_path, output_video_file_path = self._setup_output_video(video_path, width, height, fps)
            
            # Process frames
            all_detections = []
            frame_count = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Detect humans
                human_detections = detection_service._detect_humans_yolo(frame)
                
                # Apply tracking (prefer StrongSORT if available)
                if self.strongsort_service is not None:
                    self.logger.debug(f"Using StrongSORT for tracking {len(human_detections)} detections")
                    tracked_humans = self.strongsort_service.update(human_detections, frame)
                else:
                    self.logger.debug(f"Using fallback tracking service for {len(human_detections)} detections")
                    tracked_humans = tracking_service.update(human_detections, frame)
                
                # Face recognition (only when good features available)
                face_recognitions = {}
                if face_recognition_service and human_detections:
                    face_recognitions = face_recognition_service.recognize_faces_in_detections(frame, human_detections)
                    
                    # Send Kafka alerts for recognized persons
                    if kafka_service and face_recognitions:
                        self._send_recognition_alerts(kafka_service, face_recognitions, frame_count)
                
                # Create proper mapping between detections and tracks for face recognition
                detection_to_track_mapping = self._create_detection_track_mapping(human_detections, tracked_humans)
                
                # Store tracking data in database
                if database_service:
                    for i, detection in enumerate(human_detections):
                        track = tracked_humans[i] if i < len(tracked_humans) else {}
                        track_id = track.get('track_id', 0)
                        track_age = track.get('age', 0)
                        track_hits = track.get('hits', 0)
                        track_state = track.get('state', 'Unknown')
                        database_service.store_tracking_data(
                            track_id=track_id,
                            person_id=f'human_{track_id}',
                            frame_id=frame_count,
                            bbox=detection['bbox'],
                            confidence=detection['confidence'],
                            track_age=track_age,
                            track_hits=track_hits,
                            track_state=track_state,
                            timestamp=frame_count / fps,
                            source=video_path
                        )
                
                # Annotate frame
                if self.save_annotated_video:
                    annotated_frame = self._annotate_frame_with_humans(
                        frame, human_detections, tracked_humans, face_recognitions
                    )
                    output_video_path.write(annotated_frame)
                
                # Store detection data
                frame_detections = []
                for i, detection in enumerate(human_detections):
                    # Get the correct track ID using the mapping
                    track_id = detection_to_track_mapping.get(i, 'N/A')
                    if isinstance(track_id, np.integer):
                        track_id = int(track_id)
                    
                    # Get display_id from tracked_humans if available (for consistent UI display)
                    display_id = 'N/A'
                    if i < len(tracked_humans):
                        display_id = tracked_humans[i].get('display_id', tracked_humans[i].get('persistent_id', 'N/A'))
                        if isinstance(display_id, np.integer):
                            display_id = int(display_id)
                    
                    # Get persistent_id for internal tracking
                    persistent_id = 'N/A'
                    if i < len(tracked_humans):
                        persistent_id = tracked_humans[i].get('persistent_id', 'N/A')
                        if isinstance(persistent_id, np.integer):
                            persistent_id = int(persistent_id)
                    
                    # Get person name and status using persistent cache
                    person_name = 'Human'
                    person_status = 'normal'
                    person_label = 'customer'  # Default to customer
                    match_confidence = 0.0
                    
                    # First, try to get persistent name from cache (if enabled)
                    if (self.name_persistence_enabled and display_id != 'N/A' and 
                        isinstance(display_id, (int, np.integer))):
                        cached_name, cached_status, cached_label = self._get_track_name(int(display_id), frame_count)
                        if cached_name:
                            person_name = cached_name
                            person_status = cached_status
                            person_label = cached_label
                    
                    # Update cache with new recognition results if available
                    # Use the detection index to get face recognition results
                    if i in face_recognitions:
                        recognition_data = face_recognitions[i]
                        new_name = recognition_data['person_name']
                        new_confidence = recognition_data.get('confidence', 0.0)
                        new_status = recognition_data.get('status', 'normal')
                        new_label = recognition_data.get('person_label', 'customer')
                        match_confidence = new_confidence
                        
                        # Update cache with new recognition (if persistence enabled)
                        if (self.name_persistence_enabled and display_id != 'N/A' and 
                            isinstance(display_id, (int, np.integer))):
                            self._update_track_name_cache(int(display_id), new_name, new_confidence, frame_count, new_label)
                            # Use the new recognition if confidence is high enough
                            if new_confidence > self.name_confidence_threshold:
                                person_name = new_name
                                person_status = new_status
                                person_label = new_label
                    
                    detection_data = {
                        'frame_id': int(frame_count),
                        'detection_id': int(i),
                        'track_id': display_id,  # Use display_id for consistent UI display
                        'strongsort_track_id': track_id,  # Keep original StrongSORT track_id for reference
                        'persistent_id': persistent_id,  # Internal persistent_id for re-identification
                        'display_id': display_id,  # Explicit display_id field for UI
                        'person_id': (f"human_{display_id}" if display_id != 'N/A' else 'Unknown'),
                        'person_name': person_name,
                        'status': person_status,
                        'label': person_label,  # Employee or Customer label
                        'match_confidence': float(match_confidence),
                        'bbox_x1': float(detection['bbox'][0]),
                        'bbox_y1': float(detection['bbox'][1]),
                        'bbox_x2': float(detection['bbox'][2]),
                        'bbox_y2': float(detection['bbox'][3]),
                        'confidence': float(detection['confidence']),
                        'track_age': int(self._get_track_info(tracked_humans, track_id, 'age', 0)),
                        'track_hits': int(self._get_track_info(tracked_humans, track_id, 'hits', 0)),
                        'track_state': self._get_track_info(tracked_humans, track_id, 'state', 'Unknown'),
                        'timestamp': float(frame_count / fps),
                        'source': str(video_path)
                    }
                    frame_detections.append(detection_data)
                
                all_detections.extend(frame_detections)
                
                # Clean up old track names
                current_track_ids = {int(track_id) for track_id in detection_to_track_mapping.values() 
                                   if track_id != 'N/A' and isinstance(track_id, (int, np.integer))}
                self._cleanup_old_track_names(current_track_ids, frame_count)

                # Stream to Kafka if available
                if kafka_service:
                    for d in frame_detections:
                        kafka_service.send_detection(d)
                    if frame_detections:
                        self.logger.info(f"Sent {len(frame_detections)} face detections to Kafka")
                    else:
                        self.logger.debug("No face detections to send to Kafka")
                else:
                    self.logger.warning("Kafka service not available - face detection data not being streamed")
                processed_frames += 1
                frame_count += 1
                
                # Show progress
                if self.show_progress and frame_count % 30 == 0:
                    self.logger.info(f"Processed {frame_count}/{total_frames} frames")
                
                # Limit frames if specified
                if self.max_frames and processed_frames >= self.max_frames:
                    break
            
            cap.release()
            if output_video_path:
                output_video_path.release()
            
            # Save CSV if requested
            csv_path = None
            if self.save_csv and all_detections:
                csv_path = self._save_detections_csv(all_detections, video_path)
                # Also append to global face stream CSV for real-time consumers
                if self.global_stream_csv_path:
                    try:
                        self._append_to_global_stream_csv(all_detections, self.global_stream_csv_path)
                    except Exception as e:
                        logger.error(f"Failed to append to global stream CSV: {e}")
            
            return {
                'success': True,
                'total_frames': frame_count,
                'processed_frames': processed_frames,
                'total_detections': len(all_detections),
                'output_video': str(output_video_file_path) if self.save_annotated_video else None,
                'csv_file': csv_path
            }
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {'success': False, 'error': str(e)}
    
    def _setup_output_video(self, video_path: str, width: int, height: int, fps: float) -> Tuple[cv2.VideoWriter, str]:
        """Setup output video writer and return writer and file path"""
        output_path = self.output_path / f"{Path(video_path).stem}_processed.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        return writer, str(output_path)

    def _append_to_global_stream_csv(self, detections: List[Dict[str, Any]], stream_csv_path: str) -> None:
        """Append detections to the global face stream CSV used by API layer"""
        stream_path = Path(stream_csv_path)
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure header exists
        header = [
            'frame_id', 'detection_id', 'track_id', 'strongsort_track_id', 'persistent_id', 'display_id',
            'person_id', 'person_name', 'match_confidence', 'status', 'label', 'bbox_x1', 'bbox_y1', 
            'bbox_x2', 'bbox_y2', 'confidence', 'track_age', 'track_hits', 'track_state', 
            'timestamp', 'source'
        ]
        file_exists = stream_path.exists()
        with open(stream_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerows(detections)
    
    def _annotate_frame_with_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        recognitions: List[Dict[str, Any]],
        tracked_faces: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Annotate frame with face detections and recognitions"""
        annotated = frame.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Get recognition info
            recognition = recognitions[i] if i < len(recognitions) else {}
            person_name = recognition.get('person_name', 'Unknown')
            is_known = recognition.get('is_known', False)
            track_id = tracked_faces[i].get('track_id', 'N/A') if i < len(tracked_faces) else 'N/A'
            track_state = tracked_faces[i].get('state', 'Unknown') if i < len(tracked_faces) else 'Unknown'
            
            # Choose color based on recognition status
            if is_known:
                color = self.box_colors['face_known']
            else:
                color = self.box_colors['face_unknown']
            
            # Create label
            status = recognition.get('status') if i < len(recognitions) else None
            status_tag = f" [BLOCKED]" if (status is not None and str(status).lower() == 'blocked') else ""
            label = f"ID:{track_id} {person_name}{status_tag} ({confidence:.2f}) {track_state}"
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        return annotated
    
    def _send_recognition_alerts(self, kafka_service, face_recognitions: Dict[int, Dict[str, Any]], frame_count: int) -> None:
        """Send Kafka alerts for recognized persons"""
        try:
            for detection_idx, recognition_data in face_recognitions.items():
                person_name = recognition_data['person_name']
                person_status = recognition_data.get('status', 'normal')
                person_label = recognition_data.get('person_label', 'customer')
                confidence = recognition_data['confidence']
                
                # Create alert message based on status
                if person_status == 'blocked':
                    alert_message = f"Suspected Person Walking in the premises: {person_name}"
                    alert_type = "SECURITY_ALERT"
                    alert_level = "HIGH"
                    topic_type = "security_alert"
                else:
                    alert_message = f" {person_name} is crossing"
                    alert_type = "PERSON_DETECTED"
                    alert_level = "INFO"
                    topic_type = "person_recognition"
                
                # Create alert data in the format expected by the API
                alert_data = {
                    "type": "alert",  # API expects this specific type
                    "level": alert_level,
                    "message": alert_message,
                    "person_name": person_name,
                    "person_status": person_status,
                    "person_label": person_label,
                    "confidence": confidence,
                    "frame_id": frame_count,
                    "detection_id": detection_idx,
                    "track_id": detection_idx,  # Add track_id for API compatibility
                    "timestamp": time.time(),
                    "location": "premises",
                    "topic_type": topic_type,
                    "alert_type": alert_type  # Keep original type for reference
                }
                
                # Send to Kafka with proper topic routing
                try:
                    # Try to send to specific topic if the service supports it
                    if hasattr(kafka_service, 'send_to_topic'):
                        kafka_service.send_to_topic(topic_type, alert_data)
                    else:
                        # Fallback to standard send_detection
                        kafka_service.send_detection(alert_data)
                    
                    self.logger.info(f"✅ Sent {alert_type} alert to Kafka: {alert_message}")
                except Exception as kafka_error:
                    self.logger.error(f"Kafka send error: {kafka_error}")
                    # Try direct producer if available
                    if hasattr(kafka_service, 'producer'):
                        try:
                            kafka_service.producer.send(topic_type, alert_data)
                            self.logger.info(f"✅ Sent {alert_type} alert via direct producer: {alert_message}")
                        except Exception as direct_error:
                            self.logger.error(f"Direct producer error: {direct_error}")
                
        except Exception as e:
            self.logger.error(f"Error sending recognition alerts: {e}")
    
    def _create_detection_track_mapping(self, human_detections, tracked_humans):
        """Create proper mapping between detection indices and track IDs."""
        mapping = {}
        
        # If we have StrongSORT, use its internal mapping
        if self.strongsort_service is not None:
            # StrongSORT returns tracks in the same order as detections
            for i, track in enumerate(tracked_humans):
                if i < len(human_detections):
                    track_id = track.get('track_id')
                    if track_id is not None:
                        mapping[i] = track_id
        else:
            # For regular tracking service, assume 1:1 mapping
            for i, track in enumerate(tracked_humans):
                if i < len(human_detections):
                    track_id = track.get('track_id')
                    if track_id is not None:
                        mapping[i] = track_id
        
        return mapping
    
    def _get_track_info(self, tracked_humans, track_id, key, default_value):
        """Get track information by track ID."""
        if track_id == 'N/A':
            return default_value
        
        for track in tracked_humans:
            if track.get('track_id') == track_id:
                return track.get(key, default_value)
        
        return default_value
    
    def _annotate_frame_with_humans(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        tracked_humans: List[Dict[str, Any]],
        face_recognitions: Dict[int, Dict[str, Any]] = None
    ) -> np.ndarray:
        """Annotate frame with human detections and tracking"""
        annotated = frame.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Use persistent_id if available, otherwise fall back to track_id
            persistent_id = tracked_humans[i].get('persistent_id', 'N/A') if i < len(tracked_humans) else 'N/A'
            track_id = tracked_humans[i].get('track_id', 'N/A') if i < len(tracked_humans) else 'N/A'
            track_state = tracked_humans[i].get('state', 'Unknown') if i < len(tracked_humans) else 'Unknown'
            
            # Use persistent_id for display and caching
            display_id = persistent_id if persistent_id != 'N/A' else track_id
            
            # Get person name and status using persistent cache
            person_name = ""
            person_status = "normal"
            
            # First, try to get persistent name from cache (if enabled)
            if (self.name_persistence_enabled and display_id != 'N/A' and 
                isinstance(display_id, (int, np.integer))):
                cached_name, cached_status, cached_label = self._get_track_name(int(display_id), 0)  # frame_count not available here
                if cached_name:
                    person_name = f" {cached_name}"
                    person_status = cached_status
            
            # Override with new recognition if available
            if face_recognitions and i in face_recognitions:
                recognition_data = face_recognitions[i]
                person_name = f" {recognition_data['person_name']}"
                person_status = recognition_data.get('status', 'normal')
                
                # Choose color based on status
                if person_status == 'blocked':
                    color = self.box_colors['face_unknown']  # Red for blocked persons
                else:
                    color = self.box_colors['face_known']  # Green for normal recognized faces
            else:
                # Choose color based on track state
                if track_state == 'Confirmed':
                    color = self.box_colors['track_confirmed']
                else:
                    color = self.box_colors['track_tentative']
            
            # Create label according to requirements: "ID: 20 Name" or "ID: 20 Name [BLOCKED]"
            if person_name:
                status_tag = " [BLOCKED]" if person_status == 'blocked' else ""
                label = f"ID: {display_id} {person_name}{status_tag}"
            else:
                label = f"ID: {display_id}"
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        return annotated
    
    def _save_detections_csv(self, detections: List[Dict[str, Any]], video_path: str) -> str:
        """Save detections to CSV file"""
        csv_path = self.output_path / f"{Path(video_path).stem}_detections.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'frame_id', 'detection_id', 'track_id', 'strongsort_track_id', 'persistent_id', 'display_id',
                'person_id', 'person_name', 'match_confidence', 'status', 'label', 'bbox_x1', 'bbox_y1', 
                'bbox_x2', 'bbox_y2', 'confidence', 'track_age', 'track_hits', 'track_state', 
                'timestamp', 'source'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detections)
        
        return str(csv_path)
    
    def _update_track_name_cache(self, track_id: int, person_name: str, confidence: float, frame_count: int, person_label: str = 'customer') -> None:
        """Update track name cache with persistence logic"""
        if not person_name or person_name == "Unknown":
            return
        
        # Initialize cache entry if not exists
        if track_id not in self.track_name_cache:
            self.track_name_cache[track_id] = {
                'name': person_name,
                'confidence': confidence,
                'frames_seen': 1,
                'last_seen': frame_count,
                'label': person_label
            }
            self.logger.info(f"Track {track_id} assigned name: {person_name} (confidence: {confidence:.3f})")
            return
        
        cache_entry = self.track_name_cache[track_id]
        
        # Update name only if confidence is significantly higher
        if confidence > cache_entry['confidence'] + self.name_improvement_threshold:
            cache_entry['name'] = person_name
            cache_entry['confidence'] = confidence
            cache_entry['frames_seen'] += 1
            cache_entry['last_seen'] = frame_count
            cache_entry['label'] = person_label
            self.logger.info(f"Track {track_id} name updated: {person_name} (confidence: {confidence:.3f})")
        else:
            # Just update the tracking info
            cache_entry['frames_seen'] += 1
            cache_entry['last_seen'] = frame_count
    
    def _get_track_name(self, track_id: int, frame_count: int) -> Tuple[str, str, str]:
        """Get persistent name for track with fallback logic"""
        if track_id not in self.track_name_cache:
            return "", "normal", "customer"
        
        cache_entry = self.track_name_cache[track_id]
        
        # Check if name is still valid (not too old)
        frames_since_last_seen = frame_count - cache_entry['last_seen']
        if frames_since_last_seen > self.name_persistence_frames:
            # Name is too old, remove from cache
            del self.track_name_cache[track_id]
            return "", "normal", "customer"
        
        return cache_entry['name'], "normal", cache_entry.get('label', 'customer')
    
    def _cleanup_old_track_names(self, current_track_ids: set, frame_count: int) -> None:
        """Clean up old track names from cache"""
        tracks_to_remove = []
        
        for track_id, cache_entry in self.track_name_cache.items():
            if track_id not in current_track_ids:
                frames_since_last_seen = frame_count - cache_entry['last_seen']
                if frames_since_last_seen > self.name_persistence_frames:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_name_cache[track_id]
            self.logger.debug(f"Removed old track name cache for track {track_id}")
