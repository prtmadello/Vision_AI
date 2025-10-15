"""
Unified Kafka Producer and Consumer Service
"""

import json
import time
from typing import Dict, Any, Optional, Callable
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import threading
from utils.logger import setup_logger

logger = setup_logger(__name__)


class KafkaService:
    """Unified Kafka producer and consumer service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        
        self.producer = None
        self.consumer = None
        self.consumer_thread = None
        self.running = False
        
        self._initialize_producer()
        self._initialize_consumer()
    
    def _initialize_producer(self):
        """Initialize Kafka producer"""
        try:
            bootstrap_servers = self.config.get('bootstrap_servers', ['localhost:9092'])
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=3,
                retry_backoff_ms=100
            )
            self.logger.info("Kafka producer initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka producer: {e}")
            self.producer = None
    
    def _initialize_consumer(self):
        """Initialize Kafka consumer"""
        try:
            bootstrap_servers = self.config.get('bootstrap_servers', ['localhost:9092'])
            
            # Listen to multiple topics
            topics = [
                self.config.get('topic', 'paarvai_vision'),
                self.config.get('people_tracking_topic', 'people_tracking'),
                self.config.get('person_recognition_topic', 'person_recognition'),
                self.config.get('security_alert_topic', 'security_alert')
            ]
            # Remove duplicates while preserving order
            topics = list(dict.fromkeys(topics))
            
            group_id = self.config.get('group_id', 'paarvai_api_consumer')
            
            self.consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset=self.config.get('auto_offset_reset', 'latest'),
                enable_auto_commit=self.config.get('enable_auto_commit', True),
                session_timeout_ms=self.config.get('session_timeout_ms', 30000),
                request_timeout_ms=self.config.get('request_timeout_ms', 30000)
            )
            self.logger.info(f"Kafka consumer initialized for topics: {topics}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka consumer: {e}")
            self.consumer = None
    
    def send_message(self, message: Dict[str, Any], topic: str = None) -> bool:
        """Send message to Kafka topic"""
        if not self.producer:
            self.logger.error("Producer not initialized")
            return False
        
        try:
            target_topic = topic or self.config.get('topic', 'paarvai_vision')
            future = self.producer.send(target_topic, message)
            record_metadata = future.get(timeout=10)
            self.logger.debug(f"Message sent to {record_metadata.topic} partition {record_metadata.partition}")
            return True
        except KafkaError as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    def start_consumer(self, message_handler: Callable[[Dict[str, Any]], None]):
        """Start Kafka consumer with message handler"""
        if not self.consumer:
            self.logger.error("Consumer not initialized")
            return False
        
        if self.running:
            self.logger.warning("Consumer already running")
            return True
        
        def consume_messages():
            self.running = True
            self.logger.info("Kafka consumer started")
            
            try:
                for message in self.consumer:
                    if not self.running:
                        break
                    
                    try:
                        message_handler(message.value)
                    except Exception as e:
                        self.logger.error(f"Error processing message: {e}")
                        
            except Exception as e:
                self.logger.error(f"Consumer error: {e}")
            finally:
                self.running = False
                self.logger.info("Kafka consumer stopped")
        
        self.consumer_thread = threading.Thread(target=consume_messages, daemon=True)
        self.consumer_thread.start()
        return True
    
    def stop_consumer(self):
        """Stop Kafka consumer"""
        if self.running:
            self.running = False
            if self.consumer_thread:
                self.consumer_thread.join(timeout=5)
            self.logger.info("Kafka consumer stopped")
    
    def send_alert(self, track_id: int, timestamp: float, message: str):
        """Send alert message"""
        alert_msg = {
            "type": "alert",
            "track_id": track_id,
            "timestamp": timestamp,
            "message": message
        }
        return self.send_message(alert_msg)
    
    def send_count(self, total_count: int, timestamp: float = None):
        """Send count message"""
        if timestamp is None:
            timestamp = time.time()
        
        count_msg = {
            "type": "count",
            "timestamp": timestamp,
            "total_count": total_count
        }
        return self.send_message(count_msg)
    
    def send_detection(self, detection_data: Dict[str, Any]):
        """Send detection data"""
        detection_msg = {
            "type": "detection",
            "data": detection_data,
            "timestamp": time.time()
        }
        return self.send_message(detection_msg)
    
    def send_to_topic(self, topic_type: str, data: Dict[str, Any]):
        """Send data to specific topic"""
        # Map topic types to actual topic names
        topic_mapping = {
            'security_alert': self.config.get('security_alert_topic', 'security_alert'),
            'person_recognition': self.config.get('person_recognition_topic', 'person_recognition'),
            'people_tracking': self.config.get('people_tracking_topic', 'people_tracking'),
            'detection': self.config.get('topic', 'paarvai_vision')
        }
        
        target_topic = topic_mapping.get(topic_type, self.config.get('topic', 'paarvai_vision'))
        return self.send_message(data, topic=target_topic)
    
    def is_connected(self) -> bool:
        """Check if Kafka is connected"""
        return self.producer is not None and self.consumer is not None
    
    def close(self):
        """Close Kafka connections"""
        self.stop_consumer()
        
        if self.producer:
            self.producer.close()
            self.producer = None
        
        if self.consumer:
            self.consumer.close()
            self.consumer = None
        
        self.logger.info("Kafka connections closed")
