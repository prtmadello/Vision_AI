"""
Video processing utilities for Vision AI system.
"""

import cv2
import numpy as np
from typing import Iterator, Tuple, Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Video processing utilities for Vision AI."""
    
    def __init__(self, frame_skip: int = 1, max_frames: Optional[int] = None):
        """
        Initialize video processor.
        
        Args:
            frame_skip: Number of frames to skip between processing
            max_frames: Maximum number of frames to process
        """
        self.frame_skip = frame_skip
        self.max_frames = max_frames
    
    def read_video_frames(
        self, 
        video_path: str
    ) -> Iterator[Tuple[np.ndarray, int, float]]:
        """
        Read frames from video file.
        
        Args:
            video_path: Path to video file
            
        Yields:
            Tuple of (frame, frame_number, timestamp)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"FPS: {fps}, Total frames: {total_frames}")
        
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if configured
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Check max frames limit
                if self.max_frames and processed_count >= self.max_frames:
                    break
                
                # Calculate timestamp
                timestamp = frame_count / fps if fps > 0 else 0.0
                
                yield frame, frame_count, timestamp
                
                frame_count += 1
                processed_count += 1
                
        finally:
            cap.release()
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {'error': 'Could not open video'}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': duration,
            'aspect_ratio': width / height if height > 0 else 0
        }
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        frame_skip: int = 1,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """
        Extract frames from video and save to directory.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            frame_skip: Number of frames to skip
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of saved frame file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_frames = []
        processor = VideoProcessor(frame_skip, max_frames)
        
        for frame, frame_num, timestamp in processor.read_video_frames(video_path):
            frame_filename = f"frame_{frame_num:06d}_{timestamp:.2f}s.jpg"
            frame_path = output_path / frame_filename
            
            cv2.imwrite(str(frame_path), frame)
            saved_frames.append(str(frame_path))
        
        logger.info(f"Extracted {len(saved_frames)} frames to {output_dir}")
        return saved_frames
    
    def create_video_from_frames(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        codec: str = 'mp4v'
    ) -> bool:
        """
        Create video from list of frames.
        
        Args:
            frames: List of frames
            output_path: Output video path
            fps: Frames per second
            codec: Video codec
            
        Returns:
            True if successful, False otherwise
        """
        if not frames:
            logger.error("No frames provided")
            return False
        
        try:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            logger.info(f"Video saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            return False
    
    def process_video_with_annotations(
        self,
        video_path: str,
        output_path: str,
        annotation_func: callable,
        frame_skip: int = 1,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process video with custom annotation function.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            annotation_func: Function to annotate frames
            frame_skip: Frame skip interval
            max_frames: Maximum frames to process
            
        Returns:
            Processing results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'success': False, 'error': 'Could not open video'}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if configured
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Check max frames limit
                if max_frames and processed_count >= max_frames:
                    break
                
                # Apply annotation function
                annotated_frame = annotation_func(frame, frame_count)
                
                # Write frame
                out.write(annotated_frame)
                
                frame_count += 1
                processed_count += 1
                
        finally:
            cap.release()
            out.release()
        
        return {
            'success': True,
            'processed_frames': processed_count,
            'total_frames': frame_count,
            'output_path': output_path
        }
    
    def get_frame_at_timestamp(
        self,
        video_path: str,
        timestamp: float
    ) -> Optional[np.ndarray]:
        """
        Get frame at specific timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds
            
        Returns:
            Frame at timestamp or None
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        cap.release()
        
        if ret:
            return frame
        else:
            logger.error(f"Could not get frame at timestamp {timestamp}")
            return None
    
    def create_thumbnail(
        self,
        video_path: str,
        output_path: str,
        timestamp: float = 0.0
    ) -> bool:
        """
        Create thumbnail from video.
        
        Args:
            video_path: Path to video file
            output_path: Output thumbnail path
            timestamp: Timestamp for thumbnail (seconds)
            
        Returns:
            True if successful, False otherwise
        """
        frame = self.get_frame_at_timestamp(video_path, timestamp)
        
        if frame is None:
            return False
        
        try:
            cv2.imwrite(output_path, frame)
            logger.info(f"Thumbnail saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save thumbnail: {e}")
            return False
