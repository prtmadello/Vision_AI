"""
Batch Face Processing Utility
Processes a folder of images to extract faces, vectorize them, and store in database
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from services.detection_service import DetectionService
from services.face_processing_service import FaceProcessingService
from services.vectorization_service import VectorizationService
from services.database_service import DatabaseService
from utils.config_loader import ConfigLoader
from utils.logger import get_logger


class BatchFaceProcessor:
    """Batch processor for face images in a folder"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.logger = get_logger(__name__)
        
        # Initialize services
        self.detection_service = DetectionService(
            self.config_loader.get_detection_config()
        )
        self.face_processing_service = FaceProcessingService(
            self.config_loader.get_face_processing_config()
        )
        # Share the analyzer with vectorization to avoid duplicate InsightFace init
        shared_analyzer = getattr(self.face_processing_service, 'face_analyzer', None)
        self.vectorization_service = VectorizationService(
            self.config_loader.get_vectorization_config(),
            shared_face_analyzer=shared_analyzer
        )
        self.database_service = DatabaseService(
            self.config_loader.get_storage_config()
        )
        
        # Get batch processing config
        self.batch_config = self.config_loader.get('batch_processing', {})
        self.input_folder = self.batch_config.get('input_folder', '../input/faces')
        self.similarity_threshold = self.batch_config.get('similarity_threshold', 0.75)
        self.skip_existing = self.batch_config.get('skip_existing', True)
        
    def process_folder(self, folder_path: str = None) -> Dict[str, Any]:
        """Process all images in a folder for face vectorization"""
        if folder_path is None:
            folder_path = self.input_folder
            
        if not os.path.exists(folder_path):
            self.logger.error(f"Input folder does not exist: {folder_path}")
            return {'success': False, 'error': 'Folder not found'}
        
        # Get all image files
        image_files = self._get_image_files(folder_path)
        if not image_files:
            self.logger.warning(f"No image files found in {folder_path}")
            return {'success': False, 'error': 'No images found'}
        
        self.logger.info(f"Processing {len(image_files)} images from {folder_path}")
        
        results = {
            'total_images': len(image_files),
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'new_faces': 0,
            'duplicate_faces': 0,
            'details': []
        }
        
        # Get existing embeddings for duplicate detection
        existing_embeddings = self._get_existing_embeddings()
        
        for image_file in image_files:
            try:
                result = self._process_single_image(
                    image_file, existing_embeddings
                )
                results['details'].append(result)
                
                if result['status'] == 'processed':
                    results['processed'] += 1
                    if result['is_new_face']:
                        results['new_faces'] += 1
                    else:
                        results['duplicate_faces'] += 1
                elif result['status'] == 'skipped':
                    results['skipped'] += 1
                else:
                    results['errors'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {e}")
                results['errors'] += 1
                results['details'].append({
                    'file': image_file,
                    'status': 'error',
                    'error': str(e)
                })
        
        results['success'] = True
        self.logger.info(f"Batch processing completed: {results}")
        return results
    
    def _get_image_files(self, folder_path: str) -> List[str]:
        """Get all image files from folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file in os.listdir(folder_path):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(folder_path, file))
        
        return sorted(image_files)
    
    def _get_existing_embeddings(self) -> List[Dict[str, Any]]:
        """Get existing face embeddings from database for duplicate detection"""
        try:
            embeddings = self.database_service.get_all_embeddings()
            self.logger.info(f"Found {len(embeddings)} existing embeddings for duplicate detection")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error getting existing embeddings: {e}")
            return []
    
    def _process_single_image(
        self, 
        image_path: str, 
        existing_embeddings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process a single image for face vectorization"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'file': image_path,
                    'status': 'error',
                    'error': 'Could not load image'
                }
            
            # Extract person name from filename (remove extension)
            person_name = Path(image_path).stem
            
            # Detect faces
            face_detections = self.detection_service.detect_faces(image)
            if not face_detections:
                return {
                    'file': image_path,
                    'status': 'skipped',
                    'reason': 'No faces detected'
                }
            
            # Process faces
            processed_faces = self.face_processing_service.process_face_detections(
                image, face_detections
            )
            
            if not processed_faces:
                return {
                    'file': image_path,
                    'status': 'skipped',
                    'reason': 'No valid faces processed'
                }
            
            # Use the largest face (highest confidence)
            best_face = max(processed_faces, key=lambda x: x['confidence'])
            face_crop = best_face['face_crop']
            
            # Extract embedding (try crop first, fallback to full image)
            embedding = self.vectorization_service.extract_embedding_from_crop(face_crop)
            if embedding is None:
                # Fallback: let InsightFace detect and embed from the full image
                embedding = self.vectorization_service.extract_embedding(image)
            if embedding is None:
                return {
                    'file': image_path,
                    'status': 'error',
                    'error': 'Could not extract face embedding'
                }
            
            # Check for duplicates
            is_duplicate = False
            if existing_embeddings and self.skip_existing:
                match_result = self.vectorization_service.find_best_match(
                    embedding, existing_embeddings, self.similarity_threshold
                )
                if match_result['is_known']:
                    is_duplicate = True
                    self.logger.info(f"Duplicate face found for {person_name} - matches {match_result['person_name']}")
            
            if is_duplicate:
                return {
                    'file': image_path,
                    'person_name': person_name,
                    'status': 'skipped',
                    'reason': 'Duplicate face detected',
                    'matched_person': match_result['person_name'],
                    'similarity': match_result['confidence']
                }
            
            # Store in database
            person_id = f"batch_{person_name}_{os.path.basename(image_path)}"
            bbox = best_face['original_bbox']
            
            success = self.database_service.store_face_vector(
                person_id=person_id,
                person_name=person_name,
                embedding=embedding,
                bbox=bbox,
                confidence=best_face['confidence'],
                image_path=image_path,
                face_crop_path=None,  # Could save crop if needed
                status='active',
                location='batch_import'
            )
            
            if success:
                # Add to existing embeddings for future duplicate detection
                existing_embeddings.append({
                    'person_id': person_id,
                    'person_name': person_name,
                    'embedding': embedding,
                    'confidence': best_face['confidence']
                })
                
                return {
                    'file': image_path,
                    'person_name': person_name,
                    'person_id': person_id,
                    'status': 'processed',
                    'is_new_face': True,
                    'confidence': best_face['confidence']
                }
            else:
                return {
                    'file': image_path,
                    'status': 'error',
                    'error': 'Failed to store in database'
                }
                
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return {
                'file': image_path,
                'status': 'error',
                'error': str(e)
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed faces"""
        try:
            stats = self.database_service.get_face_database_stats()
            return stats
        except Exception as e:
            self.logger.error(f"Error getting processing stats: {e}")
            return {'error': str(e)}
    
    def cleanup_duplicates(self, similarity_threshold: float = None) -> Dict[str, Any]:
        """Remove duplicate faces from database"""
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        try:
            # Get all embeddings
            embeddings = self.database_service.get_all_embeddings()
            if len(embeddings) < 2:
                return {'success': True, 'removed': 0, 'message': 'Not enough faces for duplicate detection'}
            
            duplicates_found = []
            to_remove = []
            
            # Compare all pairs
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = self.vectorization_service.compare_embeddings(
                        embeddings[i]['embedding'],
                        embeddings[j]['embedding']
                    )
                    
                    if similarity > similarity_threshold:
                        # Keep the one with higher confidence
                        if embeddings[i]['confidence'] >= embeddings[j]['confidence']:
                            to_remove.append(embeddings[j]['person_id'])
                            duplicates_found.append({
                                'keep': embeddings[i]['person_name'],
                                'remove': embeddings[j]['person_name'],
                                'similarity': similarity
                            })
                        else:
                            to_remove.append(embeddings[i]['person_id'])
                            duplicates_found.append({
                                'keep': embeddings[j]['person_name'],
                                'remove': embeddings[i]['person_name'],
                                'similarity': similarity
                            })
            
            # Remove duplicates
            removed_count = 0
            for person_id in to_remove:
                if self.database_service.delete_person(person_id):
                    removed_count += 1
            
            return {
                'success': True,
                'removed': removed_count,
                'duplicates_found': len(duplicates_found),
                'details': duplicates_found
            }
            
        except Exception as e:
            self.logger.error(f"Error cleaning up duplicates: {e}")
            return {'success': False, 'error': str(e)}
