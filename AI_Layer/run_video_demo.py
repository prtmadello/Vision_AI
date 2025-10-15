#!/usr/bin/env python3
"""
Orchestrated video demo:
- Loads DB embeddings
- Runs face detection + recognition (labels include [BLOCKED] when status=='blocked')
- Runs tracking and streams detections to Kafka and global CSV
"""

import sys
from pathlib import Path

from core_ai_service import CoreAIService
from utils.config_loader import ConfigLoader


def main():
    ai = CoreAIService('config.json')

    # Load reference embeddings from DB
    refs = ai.get_all_embeddings()
    print(f"Embeddings loaded: {len(refs)}")

    # Kafka from API Layer config
    api_layer_path = Path.cwd().parent / 'API_Layer'
    sys.path.insert(0, str(api_layer_path.resolve()))
    try:
        # Import directly from the API Layer services directory
        import importlib.util
        kafka_service_path = api_layer_path / 'services' / 'kafka_service.py'
        config_loader_path = api_layer_path / 'utils' / 'config_loader.py'
        
        # Load kafka_service module
        spec = importlib.util.spec_from_file_location("kafka_service", kafka_service_path)
        kafka_service_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kafka_service_module)
        KafkaService = kafka_service_module.KafkaService
        
        # Load config_loader module
        spec = importlib.util.spec_from_file_location("config_loader", config_loader_path)
        config_loader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_loader_module)
        APIConfigLoader = config_loader_module.APIConfigLoader
        
        api_cfg = APIConfigLoader(str(api_layer_path / 'config.json'))
        kafka = KafkaService(api_cfg.get_kafka_config())
        print("✅ Kafka service initialized successfully!")
    except Exception as e:
        print(f"Kafka service import failed ({e}); trying fallback Kafka producer")
        kafka = None
        try:
            from kafka import KafkaProducer
            import json
            # Minimal shim to match send_detection interface
            class SimpleKafka:
                def __init__(self, servers, topic):
                    self.topic = topic
                    self.producer = KafkaProducer(
                        bootstrap_servers=servers,
                        value_serializer=lambda v: json.dumps(v).encode('utf-8')
                    )
                def send_detection(self, detection_data):
                    self.producer.send(self.topic, {
                        "type": "detection",
                        "data": detection_data
                    })
                def send_to_topic(self, topic_type, data):
                    """Send data to specific topic - for alerts"""
                    # Send alerts directly to the main topic
                    self.producer.send(self.topic, data)
                    self.producer.flush()
            # Load topic/servers from API config JSON directly
            import json as _json
            cfg = _json.loads((api_layer_path / 'config.json').read_text())
            servers = cfg.get('kafka', {}).get('bootstrap_servers', ['localhost:9092'])
            topic = cfg.get('kafka', {}).get('topic', 'paarvai_vision')
            kafka = SimpleKafka(servers, topic)
        except Exception as e2:
            print(f"Kafka fallback unavailable ({e2}); proceeding without Kafka streaming")
            kafka = None

    # Video path (absolute)
    video_path = str((Path(__file__).parent / 'input' / 'videos' / 'school_stream_1.mp4').resolve())
    print(f"Processing video (faces): {video_path}")

    result = ai.video_processing_service.process_video_with_face_detection(
        video_path=video_path,
        detection_service=ai.detection_service,
        face_processing_service=ai.face_processing_service,
        vectorization_service=ai.vectorization_service,
        tracking_service=ai.tracking_service,
        database_service=ai.database_service,
        reference_embeddings=refs,
        kafka_service=kafka
    )
    print("Face pipeline result:", result)

    print(f"Processing video (humans): {video_path}")
    human_result = ai.video_processing_service.process_video_with_human_detection(
        video_path=video_path,
        detection_service=ai.detection_service,
        tracking_service=ai.tracking_service,
        database_service=ai.database_service,
        kafka_service=kafka
    )
    print("Human pipeline result:", human_result)


if __name__ == '__main__':
    main()


