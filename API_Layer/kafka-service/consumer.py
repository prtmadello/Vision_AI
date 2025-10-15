import json
from kafka import KafkaConsumer


def main():
    topic = "test_topic"
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=["localhost:9092"],
        group_id="test_consumer_group",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    print(f"Consuming messages from topic '{topic}' (Ctrl+C to stop)...")
    try:
        for msg in consumer:
            print("received:", msg.value)
    except KeyboardInterrupt:
        print("\nStopping consumer...")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()


