"""
Data Stream Service - CSV and Kafka data processing
"""

import pandas as pd
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataStreamService:
    """Service for processing data streams from CSV and Kafka"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        
        self.csv_path = Path(self.config.get('csv_path', 'output/face_stream.csv'))
        self.poll_interval = self.config.get('poll_interval', 0.1)
        self.batch_size = self.config.get('batch_size', 100)
        
        self.running = False
        self.stream_thread = None
        self.last_position = 0
        
        self.monitors = {}
        self.alert_threshold = self.config.get('alert_threshold', 0.7)
        self.inactivity_threshold = self.config.get('inactivity_threshold', 5.0)
        self.polygon_points = self.config.get('polygon_points', [])
        
        # Line crossing configuration
        self.line_crossing_enabled = self.config.get('line_crossing_enabled', True)
        self.entry_line_y = self.config.get('entry_line_y', 300)  # Y coordinate for entry line
        self.exit_line_y = self.config.get('exit_line_y', 500)    # Y coordinate for exit line
        self.track_positions = {}  # Track previous positions for line crossing detection
        
        # Person recognition cache
        self.person_cache = {}  # Cache person info by track_id
        self.recognition_enabled = self.config.get('recognition_enabled', True)
        
        # Alert deduplication for long-standing alerts
        self.person_alert_history = {}  # person_name -> last_alert_time for long-standing alerts
        self.alert_cooldown = 300  # 5 minutes cooldown for same person long-standing alerts
    
    def start_csv_monitoring(self, message_handler: Callable[[Dict[str, Any]], None]):
        """Start monitoring CSV file for new data"""
        if self.running:
            self.logger.warning("CSV monitoring already running")
            return True
        
        def monitor_csv():
            self.running = True
            self.logger.info("CSV monitoring started")
            
            try:
                while self.running:
                    if self.csv_path.exists():
                        self._process_new_csv_data(message_handler)
                    else:
                        self.logger.debug(f"CSV file not found: {self.csv_path}")
                    
                    time.sleep(self.poll_interval)
            except Exception as e:
                self.logger.error(f"CSV monitoring error: {e}")
            finally:
                self.running = False
                self.logger.info("CSV monitoring stopped")
        
        self.stream_thread = threading.Thread(target=monitor_csv, daemon=True)
        self.stream_thread.start()
        return True
    
    def stop_csv_monitoring(self):
        """Stop CSV monitoring"""
        if self.running:
            self.running = False
            if self.stream_thread:
                self.stream_thread.join(timeout=5)
            self.logger.info("CSV monitoring stopped")
    
    def process_detection_record(self, data: Dict[str, Any], message_handler: Callable[[Dict[str, Any]], None]):
        """Public method: process a detection record dict and emit derived messages.
        Expects keys consistent with AI Layer CSV columns (frame_id, detection_id, track_id, person_id,
        person_name, match_confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, track_age, track_hits, timestamp).
        """
        try:
            if not isinstance(data, dict):
                return
            required_keys = [
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'track_id', 'timestamp'
            ]
            if not all(k in data for k in required_keys):
                return
            self._process_detection_data(data, message_handler)
        except Exception as e:
            self.logger.error(f"Error processing detection record: {e}")
    
    def _process_new_csv_data(self, message_handler: Callable[[Dict[str, Any]], None]):
        """Process new data from CSV file"""
        try:
            with open(self.csv_path, 'r') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
            
            if not new_lines:
                return
            
            for line in new_lines:
                if not line.strip():
                    continue
                
                try:
                    row_data = self._parse_csv_line(line)
                    if row_data:
                        self._process_detection_data(row_data, message_handler)
                except Exception as e:
                    self.logger.error(f"Error processing CSV line: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {e}")
    
    def _parse_csv_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse CSV line into structured data"""
        try:
            parts = line.strip().split(',')
            if len(parts) < 13:
                return None
            
            return {
                'frame_id': int(parts[0]) if parts[0].isdigit() else 0,
                'detection_id': int(parts[1]) if parts[1].isdigit() else 0,
                'track_id': int(parts[2]) if parts[2].isdigit() else 0,
                'person_id': parts[3] if parts[3] else 'Unknown',
                'person_name': parts[4] if parts[4] else 'Unknown',
                'match_confidence': float(parts[5]) if parts[5].replace('.', '').isdigit() else 0.0,
                'bbox_x1': float(parts[6]) if parts[6].replace('.', '').isdigit() else 0.0,
                'bbox_y1': float(parts[7]) if parts[7].replace('.', '').isdigit() else 0.0,
                'bbox_x2': float(parts[8]) if parts[8].replace('.', '').isdigit() else 0.0,
                'bbox_y2': float(parts[9]) if parts[9].replace('.', '').isdigit() else 0.0,
                'track_age': int(parts[10]) if parts[10].isdigit() else 0,
                'track_hits': int(parts[11]) if parts[11].isdigit() else 0,
                'timestamp': float(parts[12]) if parts[12].replace('.', '').isdigit() else time.time()
            }
        except Exception as e:
            self.logger.error(f"Error parsing CSV line: {e}")
            return None
    
    def _process_detection_data(self, data: Dict[str, Any], message_handler: Callable[[Dict[str, Any]], None]):
        """Process detection data and check for alerts and line crossings"""
        track_id = data['track_id']
        timestamp = data['timestamp']
        
        # Check if person is inside polygon
        inside_polygon = self._is_inside_polygon(data['bbox_x1'], data['bbox_y1'], data['bbox_x2'], data['bbox_y2'])
        
        # Check for line crossings
        if self.line_crossing_enabled:
            self._check_line_crossing(track_id, data, message_handler)
        
        # Update monitor for this track
        if track_id not in self.monitors:
            self.monitors[track_id] = {
                'entry_time': timestamp,
                'last_seen': timestamp,
                'alert_sent': False,
                'inside_box': inside_polygon
            }
        
        monitor = self.monitors[track_id]
        monitor['last_seen'] = timestamp
        monitor['inside_box'] = inside_polygon
        
        # Check for alerts
        if inside_polygon and not monitor['alert_sent']:
            if (timestamp - monitor['entry_time']) >= self.alert_threshold:
                monitor['alert_sent'] = True
                person_info = self._get_person_info(track_id, data)
                person_name = person_info['person_name']
                
                # Check if we should send this long-standing alert (deduplication)
                should_send = True
                if person_name and person_name.strip():
                    person_name_lower = person_name.strip().lower()
                    current_time = timestamp
                    
                    if person_name_lower in self.person_alert_history:
                        last_alert_time = self.person_alert_history[person_name_lower]
                        time_since_last_alert = current_time - last_alert_time
                        
                        if time_since_last_alert < self.alert_cooldown:
                            should_send = False
                            self.logger.info(f"Suppressing duplicate long-standing alert for {person_name} "
                                           f"(last sent {time_since_last_alert:.1f}s ago)")
                    
                    if should_send:
                        # Update the last alert time for this person
                        self.person_alert_history[person_name_lower] = current_time
                
                if should_send:
                    alert_msg = {
                        "type": "alert",
                        "track_id": track_id,
                        "person_id": person_info['person_id'],
                        "person_name": person_name,
                        "person_status": person_info['status'],
                        "recognition_type": person_info['recognition_type'],
                        "match_confidence": person_info['match_confidence'],
                        "detection_confidence": person_info['detection_confidence'],
                        "timestamp": timestamp,
                        "message": f"{person_info['track_id']} stayed more than {self.alert_threshold}s inside the polygon!"
                    }
                    message_handler(alert_msg)
                    self.logger.info(f"ALERT: Ref ID : {person_info['track_id']} stayed more than {self.alert_threshold}s inside the polygon!")
        
        # Clean up inactive monitors
        inactive_ids = [tid for tid, m in self.monitors.items() 
                       if timestamp - m['last_seen'] > self.inactivity_threshold]
        for tid in inactive_ids:
            del self.monitors[tid]
            # Also clean up track positions
            if tid in self.track_positions:
                del self.track_positions[tid]
        
        # Send count update
        total_inside = sum(1 for m in self.monitors.values() if m['inside_box'])
        count_msg = {
            "type": "count",
            "timestamp": timestamp,
            "total_inside": total_inside
        }
        message_handler(count_msg)
    
    def _is_inside_polygon(self, bbox_x1: float, bbox_y1: float, bbox_x2: float, bbox_y2: float) -> bool:
        """Check if bounding box center is inside polygon"""
        if not self.polygon_points:
            return True
        
        cx = (bbox_x1 + bbox_x2) / 2
        cy = bbox_y2  # Use bottom center
        
        return self._point_in_polygon(cx, cy, self.polygon_points)
    
    def _point_in_polygon(self, cx: float, cy: float, polygon: List[List[float]]) -> bool:
        """Check if point is inside polygon"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if min(p1y, p2y) < cy <= max(p1y, p2y):
                if p1y != p2y:
                    xints = (cy - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if cx < xints:
                    inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _get_person_info(self, track_id: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get person information including name and status from detection data"""
        try:
            # Check if we already have cached info for this track
            if track_id in self.person_cache:
                cached_info = self.person_cache[track_id]
                # Update last seen time
                cached_info['last_seen'] = data.get('timestamp', 0)
                return cached_info
            
            # Extract person info from detection data
            person_id = data.get('person_id', 'Unknown')
            person_name = data.get('person_name', 'Unknown')
            match_confidence = data.get('match_confidence', 0.0)
            status = data.get('status', 'unknown')
            person_label = data.get('person_label', 'customer')  # Default to customer
            
            # Enhanced logic for unified tracking + recognition
            if person_id != 'Unknown' and match_confidence > 0.5:
                # Face recognition successful
                final_status = 'known'
                final_name = person_name if person_name != 'Unknown' else person_id
                recognition_type = 'face_recognized'
            elif person_id.startswith('human_'):
                # Human detection only (no face recognition)
                final_status = 'tracked'
                final_name = f"Person {track_id}"
                recognition_type = 'human_detected'
            else:
                # Unknown person
                final_status = 'unknown'
                final_name = 'Unknown Person'
                recognition_type = 'unknown'
            
            person_info = {
                'person_id': person_id,
                'person_name': final_name,
                'status': final_status,
                'person_label': person_label,
                'recognition_type': recognition_type,
                'match_confidence': match_confidence,
                'track_id': track_id,
                'last_seen': data.get('timestamp', 0),
                'detection_confidence': data.get('confidence', 0.0)
            }
            
            # Cache the person info
            self.person_cache[track_id] = person_info
            
            return person_info
            
        except Exception as e:
            self.logger.error(f"Error getting person info for track {track_id}: {e}")
            return {
                'person_id': 'Unknown',
                'person_name': 'Unknown Person',
                'status': 'unknown',
                'recognition_type': 'error',
                'match_confidence': 0.0,
                'track_id': track_id,
                'last_seen': data.get('timestamp', 0),
                'detection_confidence': 0.0
            }
    
    def _check_line_crossing(self, track_id: Any, data: Dict[str, Any], message_handler: Callable[[Dict[str, Any]], None]):
        """Check if person crossed entry or exit lines"""
        try:
            # Get current position (bottom center of bounding box)
            current_y = data['bbox_y2']  # Bottom of bounding box
            current_x = (data['bbox_x1'] + data['bbox_x2']) / 2  # Center X
            
            # Get previous position
            if track_id in self.track_positions:
                prev_y = self.track_positions[track_id]['y']
                prev_x = self.track_positions[track_id]['x']
                
                # Check for entry line crossing (moving from above entry line to below)
                if prev_y < self.entry_line_y and current_y >= self.entry_line_y:
                    # Person entered (moved down past entry line)
                    person_info = self._get_person_info(track_id, data)
                    line_event = {
                        "type": "line_event",
                        "event_type": "entry",
                        "person_id": person_info['person_id'],
                        "person_name": person_info['person_name'],
                        "person_status": person_info['status'],
                        "recognition_type": person_info['recognition_type'],
                        "match_confidence": person_info['match_confidence'],
                        "detection_confidence": person_info['detection_confidence'],
                        "track_id": track_id,
                        "timestamp": data['timestamp'],
                        "frame_id": data.get('frame_id', 0),
                        "position": {"x": current_x, "y": current_y},
                        "line_y": self.entry_line_y
                    }
                    message_handler(line_event)
                    self.logger.info(f"LINE EVENT: {person_info['person_name']} ({person_info['status']}, {person_info['recognition_type']}) ENTERED at y={current_y}")
                
                # Check for exit line crossing (moving from below exit line to above)
                elif prev_y > self.exit_line_y and current_y <= self.exit_line_y:
                    # Person exited (moved up past exit line)
                    person_info = self._get_person_info(track_id, data)
                    line_event = {
                        "type": "line_event",
                        "event_type": "exit",
                        "person_id": person_info['person_id'],
                        "person_name": person_info['person_name'],
                        "person_status": person_info['status'],
                        "recognition_type": person_info['recognition_type'],
                        "match_confidence": person_info['match_confidence'],
                        "detection_confidence": person_info['detection_confidence'],
                        "track_id": track_id,
                        "timestamp": data['timestamp'],
                        "frame_id": data.get('frame_id', 0),
                        "position": {"x": current_x, "y": current_y},
                        "line_y": self.exit_line_y
                    }
                    message_handler(line_event)
                    self.logger.info(f"LINE EVENT: {person_info['person_name']} ({person_info['status']}, {person_info['recognition_type']}) EXITED at y={current_y}")
            
            # Update position for next frame
            self.track_positions[track_id] = {
                'x': current_x,
                'y': current_y,
                'timestamp': data['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"Error checking line crossing for track {track_id}: {e}")
    
    def get_csv_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent data from CSV file"""
        if not self.csv_path.exists():
            return []
        
        try:
            df = pd.read_csv(self.csv_path)
            if len(df) > limit:
                df = df.tail(limit)
            return df.to_dict('records')
        except Exception as e:
            self.logger.error(f"Error reading CSV data: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data stream statistics"""
        return {
            'csv_path': str(self.csv_path),
            'csv_exists': self.csv_path.exists(),
            'monitoring_active': self.running,
            'active_monitors': len(self.monitors),
            'poll_interval': self.poll_interval,
            'batch_size': self.batch_size
        }
    
    def is_running(self) -> bool:
        """Check if data stream is running"""
        return self.running
