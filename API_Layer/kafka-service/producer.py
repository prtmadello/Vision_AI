import json
import time
from pathlib import Path
from kafka import KafkaProducer


def parse_csv_line(line: str):
    parts = line.strip().split(',')
    if len(parts) < 14:  # Updated minimum length for new label column
        return None
    def to_int(s):
        try:
            return int(s)
        except Exception:
            return 0
    def to_float(s):
        try:
            return float(s)
        except Exception:
            return 0.0
    return {
        'frame_id': to_int(parts[0]),
        'detection_id': to_int(parts[1]),
        'track_id': to_int(parts[2]),
        'strongsort_track_id': to_int(parts[3]) if len(parts) > 3 else 0,
        'persistent_id': to_int(parts[4]) if len(parts) > 4 else 0,
        'person_id': parts[5] if len(parts) > 5 and parts[5] else 'Unknown',
        'person_name': parts[6] if len(parts) > 6 and parts[6] else 'Unknown',
        'match_confidence': to_float(parts[7]) if len(parts) > 7 else 0.0,
        'status': parts[8] if len(parts) > 8 and parts[8] else 'normal',
        'label': parts[9] if len(parts) > 9 and parts[9] else 'customer',
        'bbox_x1': to_float(parts[10]) if len(parts) > 10 else 0.0,
        'bbox_y1': to_float(parts[11]) if len(parts) > 11 else 0.0,
        'bbox_x2': to_float(parts[12]) if len(parts) > 12 else 0.0,
        'bbox_y2': to_float(parts[13]) if len(parts) > 13 else 0.0,
        'confidence': to_float(parts[14]) if len(parts) > 14 else 0.0,
        'track_age': to_int(parts[15]) if len(parts) > 15 else 0,
        'track_hits': to_int(parts[16]) if len(parts) > 16 else 0,
        'track_state': parts[17] if len(parts) > 17 and parts[17] else 'Unknown',
        'timestamp': to_float(parts[18]) if len(parts) > 18 else time.time(),
        'source': parts[19] if len(parts) > 19 and parts[19] else 'unknown'
    }


def main():
    topic = "paarvai_vision"
    csv_path = Path("/home/prithiviraj/Vision_AI/AI_Layer/output/face_stream.csv")

    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=3,
        linger_ms=5,
    )

    print(f"Producing AI detection messages to topic '{topic}' from '{csv_path}' (Ctrl+C to stop)...")

    # Start tailing from end of file to avoid replaying historical data
    last_position = 0
    if csv_path.exists():
        try:
            last_position = csv_path.stat().st_size
        except Exception:
            last_position = 0

    try:
        while True:
            if csv_path.exists():
                try:
                    with open(csv_path, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                except Exception:
                    new_lines = []

                for line in new_lines:
                    if not line.strip():
                        continue
                    rec = parse_csv_line(line)
                    if not rec:
                        continue
                    message = {
                        "type": "detection",
                        "data": rec
                    }
                    producer.send(topic, message)

                if new_lines:
                    producer.flush()
                    print(f"sent {len(new_lines)} detection message(s)")

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping producer...")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()


