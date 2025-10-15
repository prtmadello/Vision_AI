"""
Database service for storing face vectors and metadata
"""

import json
import psycopg2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DatabaseService:
    """Database service for face vector storage and retrieval"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        self.connection = None
        
        if self.config.get('database_enabled', True):
            self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.config.get('database_host', 'localhost'),
                port=self.config.get('database_port', 5432),
                database=self.config.get('database_name', 'madello'),
                user=self.config.get('database_user', 'postgres'),
                password=self.config.get('database_password', ''),
                sslmode=self.config.get('database_sslmode', 'prefer')
            )
            self._create_tables()
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            self.connection = None
    
    def _create_tables(self):
        """Create required tables if they don't exist"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            
            # Face vectors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_vectors (
                    id SERIAL PRIMARY KEY,
                    person_id VARCHAR(255) NOT NULL,
                    person_name VARCHAR(255),
                    image_path VARCHAR(500),
                    face_crop_path VARCHAR(500),
                    x1 INTEGER,
                    y1 INTEGER,
                    x2 INTEGER,
                    y2 INTEGER,
                    width INTEGER,
                    height INTEGER,
                    confidence_score FLOAT,
                    vector_features JSONB,
                    vector_dimension INTEGER,
                    status VARCHAR(50) DEFAULT NULL,
                    location VARCHAR(255) DEFAULT NULL,
                    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Person metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS person_metadata (
                    person_id VARCHAR(255) PRIMARY KEY,
                    person_name VARCHAR(255),
                    total_detections INTEGER DEFAULT 0,
                    average_confidence FLOAT DEFAULT 0.0,
                    status VARCHAR(50) DEFAULT NULL,
                    location VARCHAR(255) DEFAULT NULL,
                    first_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tracking data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracking_data (
                    id SERIAL PRIMARY KEY,
                    track_id INTEGER NOT NULL,
                    person_id VARCHAR(255),
                    frame_id INTEGER,
                    bbox_x1 FLOAT,
                    bbox_y1 FLOAT,
                    bbox_x2 FLOAT,
                    bbox_y2 FLOAT,
                    confidence FLOAT,
                    track_age INTEGER,
                    track_hits INTEGER,
                    track_state VARCHAR(50),
                    status VARCHAR(50) DEFAULT NULL,
                    location VARCHAR(255) DEFAULT NULL,
                    timestamp FLOAT,
                    source VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.connection.commit()
            self.logger.info("Database tables created/verified")
            
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
    
    def store_face_vector(
        self,
        person_id: str,
        person_name: str,
        embedding: np.ndarray,
        bbox: List[float],
        confidence: float,
        image_path: str = None,
        face_crop_path: str = None,
        status: str = None,
        location: str = None
    ) -> bool:
        """Store face vector in database"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            x1, y1, x2, y2 = bbox
            width = int(x2 - x1)
            height = int(y2 - y1)
            
            cursor.execute("""
                INSERT INTO face_vectors (
                    person_id, person_name, image_path, face_crop_path,
                    x1, y1, x2, y2, width, height, confidence_score,
                    vector_features, vector_dimension, status, location
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                person_id, person_name, image_path, face_crop_path,
                int(x1), int(y1), int(x2), int(y2), width, height, confidence,
                json.dumps(embedding.tolist()), len(embedding), status, location)
            )
            
            # Update person metadata
            cursor.execute("""
                INSERT INTO person_metadata (person_id, person_name, total_detections, average_confidence, status, location)
                VALUES (%s, %s, 1, %s, %s, %s)
                ON CONFLICT (person_id) 
                DO UPDATE SET 
                    total_detections = person_metadata.total_detections + 1,
                    last_detected = CURRENT_TIMESTAMP,
                    average_confidence = (person_metadata.average_confidence + %s) / 2,
                    status = COALESCE(EXCLUDED.status, person_metadata.status),
                    location = COALESCE(EXCLUDED.location, person_metadata.location)
            """, (person_id, person_name, confidence, status, location, confidence))
            
            self.connection.commit()
            self.logger.info(f"Stored face vector for person: {person_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing face vector: {e}")
            return False
    
    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all face embeddings from database"""
        if not self.connection:
            return []
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT person_id, person_name, vector_features, confidence_score, status, location
                FROM face_vectors
                ORDER BY detection_timestamp DESC
            """)
            embeddings = []
            for row in cursor.fetchall():
                person_id, person_name, vector_data, confidence, status, location = row
                # vector_data may be JSON string or already parsed (JSONB)
                if isinstance(vector_data, str):
                    try:
                        vector_list = json.loads(vector_data)
                    except Exception:
                        vector_list = []
                elif isinstance(vector_data, (list, tuple)):
                    vector_list = vector_data
                else:
                    try:
                        vector_list = json.loads(vector_data)
                    except Exception:
                        vector_list = []
                embedding = np.array(vector_list)
                embeddings.append({
                    'person_id': person_id,
                    'person_name': person_name,
                    'embedding': embedding,
                    'confidence': confidence,
                    'status': status,
                    'location': location
                })
            self.logger.info(f"Retrieved {len(embeddings)} embeddings from database")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error retrieving embeddings: {e}")
            return []
    def get_face_database_stats(self) -> Dict[str, Any]:
        """Get statistics about faces stored in the database"""
        if not self.connection:
            return {
                'total_faces': 0,
                'total_persons': 0,
                'average_confidence': 0.0,
                'database_size_mb': 0.0
            }
        try:
            cursor = self.connection.cursor()
            # Total face vectors
            cursor.execute("SELECT COUNT(*) FROM face_vectors")
            total_faces = cursor.fetchone()[0]

            # Total unique persons
            cursor.execute("SELECT COUNT(*) FROM person_metadata")
            total_persons = cursor.fetchone()[0]

            # Average confidence
            cursor.execute("SELECT COALESCE(AVG(confidence_score), 0) FROM face_vectors")
            average_confidence = float(cursor.fetchone()[0] or 0.0)

            # Database size (approx via pg_database_size if available)
            database_size_mb = 0.0
            try:
                cursor.execute("SELECT pg_database_size(%s)", (self.config.get('database_name', 'madello'),))
                size_bytes = cursor.fetchone()[0]
                database_size_mb = float(size_bytes) / (1024 * 1024)
            except Exception:
                database_size_mb = 0.0

            return {
                'total_faces': int(total_faces or 0),
                'total_persons': int(total_persons or 0),
                'average_confidence': average_confidence,
                'database_size_mb': database_size_mb
            }
        except Exception as e:
            self.logger.error(f"Error getting face database stats: {e}")
            return {'error': str(e)}

    def delete_person(self, person_id: str) -> bool:
        """Delete a person and related face vectors from the database"""
        if not self.connection:
            return False
        try:
            cursor = self.connection.cursor()
            # Delete face vectors
            cursor.execute("DELETE FROM face_vectors WHERE person_id = %s", (person_id,))
            # Delete person metadata
            cursor.execute("DELETE FROM person_metadata WHERE person_id = %s", (person_id,))
            self.connection.commit()
            self.logger.info(f"Deleted person and vectors: {person_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting person {person_id}: {e}")
            return False
    
    def store_tracking_data(
        self,
        track_id: int,
        person_id: str,
        frame_id: int,
        bbox: List[float],
        confidence: float,
        track_age: int,
        track_hits: int,
        track_state: str,
        timestamp: float,
        source: str,
        status: str = None,
        location: str = None
    ) -> bool:
        """Store tracking data in database"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            x1, y1, x2, y2 = bbox
            
            cursor.execute("""
                INSERT INTO tracking_data (
                    track_id, person_id, frame_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    confidence, track_age, track_hits, track_state, status, location, timestamp, source
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                int(track_id), person_id, int(frame_id), float(x1), float(y1), float(x2), float(y2),
                float(confidence), int(track_age), int(track_hits), track_state, status, location, float(timestamp), source
            ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing tracking data: {e}")
            return False
    
    def get_person_by_id(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get person information by ID"""
        if not self.connection:
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT person_id, person_name, total_detections, average_confidence,
                       status, location, first_detected, last_detected
                FROM person_metadata
                WHERE person_id = %s
            """, (person_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'person_id': row[0],
                    'person_name': row[1],
                    'total_detections': row[2],
                    'average_confidence': row[3],
                    'status': row[4],
                    'location': row[5],
                    'first_detected': row[6],
                    'last_detected': row[7]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving person: {e}")
            return None
    
    def update_person_status(self, person_id: str, status: str = None, location: str = None) -> bool:
        """Update person status and location"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Update person metadata
            if status is not None and location is not None:
                cursor.execute("""
                    UPDATE person_metadata 
                    SET status = %s, location = %s, last_detected = CURRENT_TIMESTAMP
                    WHERE person_id = %s
                """, (status, location, person_id))
            elif status is not None:
                cursor.execute("""
                    UPDATE person_metadata 
                    SET status = %s, last_detected = CURRENT_TIMESTAMP
                    WHERE person_id = %s
                """, (status, person_id))
            elif location is not None:
                cursor.execute("""
                    UPDATE person_metadata 
                    SET location = %s, last_detected = CURRENT_TIMESTAMP
                    WHERE person_id = %s
                """, (location, person_id))
            
            # Update face_vectors table
            if status is not None and location is not None:
                cursor.execute("""
                    UPDATE face_vectors 
                    SET status = %s, location = %s
                    WHERE person_id = %s
                """, (status, location, person_id))
            elif status is not None:
                cursor.execute("""
                    UPDATE face_vectors 
                    SET status = %s
                    WHERE person_id = %s
                """, (status, person_id))
            elif location is not None:
                cursor.execute("""
                    UPDATE face_vectors 
                    SET location = %s
                    WHERE person_id = %s
                """, (location, person_id))
            
            self.connection.commit()
            self.logger.info(f"Updated status/location for person: {person_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating person status: {e}")
            return False
    
    def get_persons_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all persons with specific status"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT person_id, person_name, total_detections, average_confidence,
                       status, location, first_detected, last_detected
                FROM person_metadata
                WHERE status = %s
                ORDER BY last_detected DESC
            """, (status,))
            
            persons = []
            for row in cursor.fetchall():
                persons.append({
                    'person_id': row[0],
                    'person_name': row[1],
                    'total_detections': row[2],
                    'average_confidence': row[3],
                    'status': row[4],
                    'location': row[5],
                    'first_detected': row[6],
                    'last_detected': row[7]
                })
            
            return persons
            
        except Exception as e:
            self.logger.error(f"Error retrieving persons by status: {e}")
            return []
    
    def get_persons_by_location(self, location: str) -> List[Dict[str, Any]]:
        """Get all persons from specific location"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT person_id, person_name, total_detections, average_confidence,
                       status, location, first_detected, last_detected
                FROM person_metadata
                WHERE location = %s
                ORDER BY last_detected DESC
            """, (location,))
            
            persons = []
            for row in cursor.fetchall():
                persons.append({
                    'person_id': row[0],
                    'person_name': row[1],
                    'total_detections': row[2],
                    'average_confidence': row[3],
                    'status': row[4],
                    'location': row[5],
                    'first_detected': row[6],
                    'last_detected': row[7]
                })
            
            return persons
            
        except Exception as e:
            self.logger.error(f"Error retrieving persons by location: {e}")
            return []
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.connection is not None
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("Database connection closed")
