"""
Video Quality Detection Service
Integrated service for the API layer
"""

import cv2
import numpy as np
from datetime import datetime
import time
import json
import os
from collections import deque
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path


class VideoQualityService:
    """Video Quality Detection Service for API integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the video quality service"""
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize data transformation layer
        self.data_transformer = DataTransformationLayer(self.config)
        
        # Initialize metrics tracking
        self.metrics = {
            'blur_score': 0, 'brightness': 0, 'obstruction_ratio': 0,
            'fps': 0, 'last_frame_time': 0, 'raw_blur': 0,
            'raw_brightness': 0, 'raw_obstruction': 0
        }
        
        # Initialize alert tracking
        self.alerts = []
        self.alert_cooldown = self.config.get('alert_cooldown', 5.0)
        self.last_alert_time = {}
        
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.last_frame_timestamp = time.time()
        
        self.logger.info("Video Quality Service initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'thresholds': {
                'BLUR_THRESHOLD': 100.0,
                'OBSTRUCTION_THRESHOLD': 0.3,
                'MIN_BRIGHTNESS': 20,
                'MAX_BRIGHTNESS': 235,
                'MIN_FPS': 15,
                'FRAME_DROP_TIMEOUT': 2.0,
                'CONSECUTIVE_FRAMES_THRESHOLD': 3
            },
            'data_transformation': {
                'temporal_smoothing': {
                    'enabled': True,
                    'window_size': 5
                },
                'adaptive_thresholds': {
                    'enabled': True,
                    'learning_rate': 0.1
                },
                'false_positive_filters': {
                    'ignore_motion_blur': True,
                    'motion_blur_threshold': 200.0,
                    'ignore_brief_obstructions': True,
                    'brief_duration_seconds': 1.0
                }
            },
            'alert_cooldown': 5.0,
            'analysis': {
                'enable_blur_detection': True,
                'enable_brightness_detection': True,
                'enable_obstruction_detection': True,
                'enable_fps_monitoring': True,
                'analyze_interval': 0.2
            }
        }
    
    def analyze_video_file(self, video_path: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Analyze a video file and return quality metrics"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count_total / fps if fps > 0 else 0
            
            # Limit frames if specified
            if max_frames and max_frames < frame_count_total:
                frame_count_total = max_frames
            
            frame_analysis = []
            alerts = []
            frame_idx = 0
            
            self.logger.info(f"Analyzing video: {video_path}")
            self.logger.info(f"Total frames: {frame_count_total}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
            
            while frame_idx < frame_count_total:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze frame
                frame_metrics = self._analyze_frame(frame)
                frame_metrics['frame_number'] = frame_idx
                frame_metrics['timestamp'] = frame_idx / fps if fps > 0 else 0
                
                # Check for alerts
                frame_alerts = self._check_thresholds(frame_metrics)
                if frame_alerts:
                    alerts.extend(frame_alerts)
                
                frame_analysis.append(frame_metrics)
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 100 == 0:
                    self.logger.info(f"Processed {frame_idx}/{frame_count_total} frames")
            
            cap.release()
            
            # Calculate summary statistics
            summary = self._calculate_summary(frame_analysis, alerts)
            
            return {
                'status': 'success',
                'video_path': video_path,
                'video_properties': {
                    'fps': fps,
                    'total_frames': frame_count_total,
                    'duration_seconds': duration
                },
                'summary': summary,
                'frame_analysis': frame_analysis,
                'alerts': alerts,
                'transformation_stats': self.data_transformer.get_stats()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing video: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'video_path': video_path
            }
    
    def analyze_video_stream(self, video_source: Any, duration_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Analyze a live video stream"""
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video source: {video_source}")
            
            frame_analysis = []
            alerts = []
            frame_idx = 0
            start_time = time.time()
            
            self.logger.info(f"Analyzing video stream: {video_source}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check duration limit
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                # Analyze frame
                frame_metrics = self._analyze_frame(frame)
                frame_metrics['frame_number'] = frame_idx
                frame_metrics['timestamp'] = time.time() - start_time
                
                # Check for alerts
                frame_alerts = self._check_thresholds(frame_metrics)
                if frame_alerts:
                    alerts.extend(frame_alerts)
                
                frame_analysis.append(frame_metrics)
                frame_idx += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
            
            cap.release()
            
            # Calculate summary statistics
            summary = self._calculate_summary(frame_analysis, alerts)
            
            return {
                'status': 'success',
                'video_source': str(video_source),
                'duration_analyzed': time.time() - start_time,
                'frames_analyzed': frame_idx,
                'summary': summary,
                'frame_analysis': frame_analysis,
                'alerts': alerts,
                'transformation_stats': self.data_transformer.get_stats()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing video stream: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'video_source': str(video_source)
            }
    
    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze a single frame"""
        current_time = time.time()
        
        # Calculate raw metrics
        raw_blur = self._calculate_blur(frame)
        raw_brightness = self._calculate_brightness(frame)
        raw_obstruction = self._detect_obstruction(frame)
        fps = self._calculate_fps()
        
        # Apply data transformation
        blur_score, brightness, obstruction_ratio = self.data_transformer.apply_temporal_smoothing(
            raw_blur, raw_brightness, raw_obstruction
        )
        
        # Update adaptive thresholds
        self.data_transformer.update_adaptive_thresholds(blur_score, brightness)
        
        return {
            'blur_score': round(blur_score, 2),
            'brightness': round(brightness, 2),
            'obstruction_ratio': round(obstruction_ratio, 2),
            'fps': round(fps, 2),
            'raw_blur': round(raw_blur, 2),
            'raw_brightness': round(raw_brightness, 2),
            'raw_obstruction': round(raw_obstruction, 2),
            'analysis_time': current_time
        }
    
    def _calculate_blur(self, frame: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        if not self.config.get('analysis', {}).get('enable_blur_detection', True):
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate brightness"""
        if not self.config.get('analysis', {}).get('enable_brightness_detection', True):
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def _detect_obstruction(self, frame: np.ndarray) -> float:
        """Detect obstruction"""
        if not self.config.get('analysis', {}).get('enable_obstruction_detection', True):
            return 0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        dark_pixels = np.sum(gray < 30)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        dark_ratio = dark_pixels / total_pixels
        
        return min(max(dark_ratio, 1 - edge_density * 5), 1.0)
    
    def _calculate_fps(self) -> float:
        """Calculate FPS"""
        if not self.config.get('analysis', {}).get('enable_fps_monitoring', True):
            return 0
        
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
            return fps
        return self.metrics['fps']
    
    def _check_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check thresholds and generate alerts"""
        alerts = []
        current_time = time.time()
        
        thresholds = self.config.get('thresholds', {})
        
        # Blur alert
        if self.data_transformer.should_alert_blur(metrics['blur_score'], thresholds.get('BLUR_THRESHOLD', 100.0)):
            alert = self._create_alert('BLUR', f'Blurry video (score: {metrics["blur_score"]:.2f})', 'warning', metrics, current_time)
            if alert:
                alerts.append(alert)
        
        # Obstruction alert
        if self.data_transformer.should_alert_obstruction(metrics['obstruction_ratio'], thresholds.get('OBSTRUCTION_THRESHOLD', 0.3)):
            alert = self._create_alert('OBSTRUCTION', f'Camera obstruction ({metrics["obstruction_ratio"]*100:.1f}%)', 'critical', metrics, current_time)
            if alert:
                alerts.append(alert)
        
        # Brightness alerts
        min_brightness = thresholds.get('MIN_BRIGHTNESS', 20)
        max_brightness = thresholds.get('MAX_BRIGHTNESS', 235)
        
        if metrics['brightness'] < min_brightness and metrics['brightness'] > 0:
            alert = self._create_alert('BRIGHTNESS', f'Video too dark ({metrics["brightness"]:.1f})', 'warning', metrics, current_time)
            if alert:
                alerts.append(alert)
        elif metrics['brightness'] > max_brightness:
            alert = self._create_alert('BRIGHTNESS', f'Video overexposed ({metrics["brightness"]:.1f})', 'warning', metrics, current_time)
            if alert:
                alerts.append(alert)
        
        # FPS alert
        if metrics['fps'] < thresholds.get('MIN_FPS', 15) and metrics['fps'] > 0:
            alert = self._create_alert('FPS', f'Low frame rate ({metrics["fps"]:.1f} FPS)', 'warning', metrics, current_time)
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _create_alert(self, alert_type: str, message: str, severity: str, metrics: Dict[str, Any], current_time: float) -> Optional[Dict[str, Any]]:
        """Create an alert if not in cooldown period"""
        if alert_type in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[alert_type]
            if time_since_last < self.alert_cooldown:
                return None
        
        self.last_alert_time[alert_type] = current_time
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity,
            'metrics': metrics.copy()
        }
        
        self.alerts.append(alert)
        return alert
    
    def _calculate_summary(self, frame_analysis: List[Dict[str, Any]], alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not frame_analysis:
            return {}
        
        # Extract metrics
        blur_scores = [f['blur_score'] for f in frame_analysis]
        brightness_values = [f['brightness'] for f in frame_analysis]
        obstruction_ratios = [f['obstruction_ratio'] for f in frame_analysis]
        fps_values = [f['fps'] for f in frame_analysis if f['fps'] > 0]
        
        # Calculate statistics
        summary = {
            'total_frames': len(frame_analysis),
            'blur': {
                'mean': round(np.mean(blur_scores), 2),
                'min': round(np.min(blur_scores), 2),
                'max': round(np.max(blur_scores), 2),
                'std': round(np.std(blur_scores), 2)
            },
            'brightness': {
                'mean': round(np.mean(brightness_values), 2),
                'min': round(np.min(brightness_values), 2),
                'max': round(np.max(brightness_values), 2),
                'std': round(np.std(brightness_values), 2)
            },
            'obstruction': {
                'mean': round(np.mean(obstruction_ratios), 2),
                'min': round(np.min(obstruction_ratios), 2),
                'max': round(np.max(obstruction_ratios), 2),
                'std': round(np.std(obstruction_ratios), 2)
            },
            'fps': {
                'mean': round(np.mean(fps_values), 2) if fps_values else 0,
                'min': round(np.min(fps_values), 2) if fps_values else 0,
                'max': round(np.max(fps_values), 2) if fps_values else 0,
                'std': round(np.std(fps_values), 2) if fps_values else 0
            },
            'alerts': {
                'total': len(alerts),
                'by_type': {},
                'by_severity': {}
            }
        }
        
        # Alert statistics
        for alert in alerts:
            alert_type = alert['type']
            severity = alert['severity']
            
            summary['alerts']['by_type'][alert_type] = summary['alerts']['by_type'].get(alert_type, 0) + 1
            summary['alerts']['by_severity'][severity] = summary['alerts']['by_severity'].get(severity, 0) + 1
        
        return summary


class DataTransformationLayer:
    """Data transformation layer to reduce false positives"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('data_transformation', {})
        self.thresholds_config = config.get('thresholds', {})
        
        window_size = self.config.get('temporal_smoothing', {}).get('window_size', 5)
        self.blur_history = deque(maxlen=window_size)
        self.brightness_history = deque(maxlen=window_size)
        self.obstruction_history = deque(maxlen=window_size)
        
        self.adaptive_blur_threshold = None
        self.adaptive_brightness_mean = None
        self.learning_rate = self.config.get('adaptive_thresholds', {}).get('learning_rate', 0.1)
        
        self.consecutive_blur_frames = 0
        self.last_obstruction_time = 0
        
        self.transformation_stats = {
            'false_positives_avoided': 0,
            'alerts_suppressed': 0,
            'smoothing_applied': 0,
            'adaptive_adjustments': 0
        }
    
    def apply_temporal_smoothing(self, blur: float, brightness: float, obstruction: float) -> tuple:
        """Apply moving average smoothing"""
        if not self.config.get('temporal_smoothing', {}).get('enabled', True):
            return blur, brightness, obstruction
        
        self.blur_history.append(blur)
        self.brightness_history.append(brightness)
        self.obstruction_history.append(obstruction)
        
        smoothed_blur = np.mean(self.blur_history) if self.blur_history else blur
        smoothed_brightness = np.mean(self.brightness_history) if self.brightness_history else brightness
        smoothed_obstruction = np.mean(self.obstruction_history) if self.obstruction_history else obstruction
        
        self.transformation_stats['smoothing_applied'] += 1
        return smoothed_blur, smoothed_brightness, smoothed_obstruction
    
    def update_adaptive_thresholds(self, blur: float, brightness: float):
        """Update adaptive thresholds"""
        if not self.config.get('adaptive_thresholds', {}).get('enabled', True):
            return
        
        if self.adaptive_blur_threshold is None:
            self.adaptive_blur_threshold = blur
            self.adaptive_brightness_mean = brightness
            return
        
        self.adaptive_blur_threshold = (
            (1 - self.learning_rate) * self.adaptive_blur_threshold +
            self.learning_rate * blur
        )
        self.adaptive_brightness_mean = (
            (1 - self.learning_rate) * self.adaptive_brightness_mean +
            self.learning_rate * brightness
        )
        self.transformation_stats['adaptive_adjustments'] += 1
    
    def should_alert_blur(self, blur: float, threshold: float) -> bool:
        """Determine if blur alert should be raised"""
        filters = self.config.get('false_positive_filters', {})
        
        if filters.get('ignore_motion_blur', True):
            motion_threshold = filters.get('motion_blur_threshold', 200.0)
            
            if blur > threshold * 0.5 and blur < motion_threshold:
                self.consecutive_blur_frames += 1
                consecutive_threshold = self.thresholds_config.get('CONSECUTIVE_FRAMES_THRESHOLD', 3)
                
                if self.consecutive_blur_frames >= consecutive_threshold:
                    self.consecutive_blur_frames = 0
                    return True
                else:
                    self.transformation_stats['false_positives_avoided'] += 1
                    return False
            else:
                self.consecutive_blur_frames = 0
        
        return blur < threshold
    
    def should_alert_obstruction(self, obstruction: float, threshold: float) -> bool:
        """Determine if obstruction alert should be raised"""
        filters = self.config.get('false_positive_filters', {})
        
        if filters.get('ignore_brief_obstructions', True):
            current_time = time.time()
            brief_duration = filters.get('brief_duration_seconds', 1.0)
            
            if obstruction > threshold:
                if self.last_obstruction_time == 0:
                    self.last_obstruction_time = current_time
                    self.transformation_stats['alerts_suppressed'] += 1
                    return False
                
                duration = current_time - self.last_obstruction_time
                if duration < brief_duration:
                    self.transformation_stats['alerts_suppressed'] += 1
                    return False
                return True
            else:
                self.last_obstruction_time = 0
                return False
        
        return obstruction > threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transformation statistics"""
        return self.transformation_stats.copy()
