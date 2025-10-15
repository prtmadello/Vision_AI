"""
File management utilities for Vision AI system.
"""

import json
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FileManager:
    """File manager for handling face crops, vectors, and metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize file manager.
        
        Args:
            config: File management configuration
        """
        self.config = config
        self.logger = logger
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.config.get('input_path', '../input'),
            self.config.get('output_path', '../output'),
            self.config.get('face_crops_path', '../output/face_crops'),
            self.config.get('vectors_path', '../output/vectors'),
            self.config.get('logs_path', '../logs'),
            self.config.get('temp_path', '../temp')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def save_face_crop(
        self,
        face_image: np.ndarray,
        filename: str,
        subdirectory: str = "",
        quality: int = 95
    ) -> Optional[str]:
        """
        Save face crop image.
        
        Args:
            face_image: Face crop image
            filename: Filename for the crop
            subdirectory: Optional subdirectory
            quality: JPEG quality (1-100)
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Create directory path
            base_path = Path(self.config.get('face_crops_path', '../output/face_crops'))
            if subdirectory:
                save_path = base_path / subdirectory
            else:
                save_path = base_path
            
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Create full file path
            file_path = save_path / filename
            
            # Save image
            if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                cv2.imwrite(str(file_path), face_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(file_path), face_image)
            
            self.logger.info(f"Face crop saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save face crop: {e}")
            return None
    
    def save_vector(
        self,
        vector: np.ndarray,
        filename: str,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Save face vector to file.
        
        Args:
            vector: Face embedding vector
            filename: Filename for the vector
            metadata: Optional metadata
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Create directory path
            vectors_path = Path(self.config.get('vectors_path', '../output/vectors'))
            vectors_path.mkdir(parents=True, exist_ok=True)
            
            # Create full file path
            file_path = vectors_path / filename
            
            # Save vector as numpy array
            np.save(str(file_path), vector)
            
            # Save metadata if provided
            if metadata:
                metadata_path = file_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Vector saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save vector: {e}")
            return None
    
    def load_vector(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load face vector from file.
        
        Args:
            file_path: Path to vector file
            
        Returns:
            Loaded vector or None if failed
        """
        try:
            vector = np.load(file_path)
            self.logger.info(f"Vector loaded: {file_path}")
            return vector
        except Exception as e:
            self.logger.error(f"Failed to load vector: {e}")
            return None
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str,
        subdirectory: str = ""
    ) -> Optional[str]:
        """
        Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            filename: Filename for metadata
            subdirectory: Optional subdirectory
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Create directory path
            base_path = Path(self.config.get('output_path', '../output'))
            if subdirectory:
                save_path = base_path / subdirectory
            else:
                save_path = base_path
            
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Create full file path
            file_path = save_path / filename
            
            # Save metadata
            with open(file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Metadata saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            return None
    
    def load_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata from JSON file.
        
        Args:
            file_path: Path to metadata file
            
        Returns:
            Loaded metadata or None if failed
        """
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            
            self.logger.info(f"Metadata loaded: {file_path}")
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            stats = {
                'face_crops': 0,
                'vectors': 0,
                'metadata_files': 0,
                'total_size_mb': 0,
                'directories': {}
            }
            
            # Count face crops
            face_crops_path = Path(self.config.get('face_crops_path', '../output/face_crops'))
            if face_crops_path.exists():
                face_crops = list(face_crops_path.rglob('*.jpg')) + list(face_crops_path.rglob('*.png'))
                stats['face_crops'] = len(face_crops)
                
                # Calculate size
                total_size = sum(f.stat().st_size for f in face_crops)
                stats['directories']['face_crops'] = {
                    'count': len(face_crops),
                    'size_mb': total_size / (1024 * 1024)
                }
                stats['total_size_mb'] += total_size / (1024 * 1024)
            
            # Count vectors
            vectors_path = Path(self.config.get('vectors_path', '../output/vectors'))
            if vectors_path.exists():
                vectors = list(vectors_path.rglob('*.npy'))
                stats['vectors'] = len(vectors)
                
                # Calculate size
                total_size = sum(f.stat().st_size for f in vectors)
                stats['directories']['vectors'] = {
                    'count': len(vectors),
                    'size_mb': total_size / (1024 * 1024)
                }
                stats['total_size_mb'] += total_size / (1024 * 1024)
            
            # Count metadata files
            output_path = Path(self.config.get('output_path', '../output'))
            if output_path.exists():
                metadata_files = list(output_path.rglob('*.json'))
                stats['metadata_files'] = len(metadata_files)
                
                # Calculate size
                total_size = sum(f.stat().st_size for f in metadata_files)
                stats['directories']['metadata'] = {
                    'count': len(metadata_files),
                    'size_mb': total_size / (1024 * 1024)
                }
                stats['total_size_mb'] += total_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting storage statistics: {e}")
            return {'error': str(e)}
    
    def cleanup_old_files(
        self,
        days_old: int = 30,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Clean up old files.
        
        Args:
            days_old: Age threshold in days
            dry_run: If True, only report what would be deleted
            
        Returns:
            Cleanup results
        """
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            files_to_delete = []
            total_size = 0
            
            # Check face crops
            face_crops_path = Path(self.config.get('face_crops_path', '../output/face_crops'))
            if face_crops_path.exists():
                for file_path in face_crops_path.rglob('*'):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            files_to_delete.append(str(file_path))
                            total_size += file_path.stat().st_size
            
            # Check vectors
            vectors_path = Path(self.config.get('vectors_path', '../output/vectors'))
            if vectors_path.exists():
                for file_path in vectors_path.rglob('*'):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            files_to_delete.append(str(file_path))
                            total_size += file_path.stat().st_size
            
            if not dry_run:
                # Actually delete files
                deleted_count = 0
                for file_path in files_to_delete:
                    try:
                        Path(file_path).unlink()
                        deleted_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to delete {file_path}: {e}")
                
                return {
                    'success': True,
                    'deleted_count': deleted_count,
                    'total_size_mb': total_size / (1024 * 1024)
                }
            else:
                return {
                    'success': True,
                    'dry_run': True,
                    'files_to_delete': len(files_to_delete),
                    'total_size_mb': total_size / (1024 * 1024),
                    'files': files_to_delete[:10]  # Show first 10 files
                }
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {'error': str(e)}
    
    def backup_data(self, backup_path: str) -> bool:
        """
        Create backup of all data.
        
        Args:
            backup_path: Path for backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup face crops
            face_crops_src = Path(self.config.get('face_crops_path', '../output/face_crops'))
            if face_crops_src.exists():
                face_crops_dst = backup_dir / "face_crops"
                shutil.copytree(face_crops_src, face_crops_dst, dirs_exist_ok=True)
            
            # Backup vectors
            vectors_src = Path(self.config.get('vectors_path', '../output/vectors'))
            if vectors_src.exists():
                vectors_dst = backup_dir / "vectors"
                shutil.copytree(vectors_src, vectors_dst, dirs_exist_ok=True)
            
            # Backup output
            output_src = Path(self.config.get('output_path', '../output'))
            if output_src.exists():
                output_dst = backup_dir / "output"
                shutil.copytree(output_src, output_dst, dirs_exist_ok=True)
            
            self.logger.info(f"Data backup created: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
