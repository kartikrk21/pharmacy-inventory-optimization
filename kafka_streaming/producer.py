import json
import time
import argparse
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError

from data_generation.prescription_generator import PrescriptionDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrescriptionKafkaProducer:
    def __init__(self, bootstrap_servers: str, topic: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic

        self.producer = KafkaProducer(
            bootstrap_servers=[self.bootstrap_servers],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=1,
        )

        self.generator = PrescriptionDataGenerator(num_medicines=500)

        logger.info(
            f"Connected to Kafka bootstrap='{self.bootstrap_servers}' topic='{self.topic}'"
        )

    def stream(self, rate_per_second: int = 5):
        delay = 1.0 / rate_per_second
        logger.info(f"Streaming prescriptions at {rate_per_second} msg/sec")

        while True:
            try:
                event = self.generator.generate_prescription()
                self.producer.send(self.topic, event)
                logger.info("Produced message")
                time.sleep(delay)
            except KafkaError as e:
                logger.error(f"Kafka error: {e}")
            except KeyboardInterrupt:
                logger.info("Producer stopped by user")
                break


def main():
    parser = argparse.ArgumentParser(description="Kafka Prescription Producer")

    parser.add_argument(
        "--topic",
        default="prescriptions",
        help="Kafka topic name",
    )

    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:29093",
        help="Kafka bootstrap servers (default: localhost:29093)",
    )

    parser.add_argument(
        "--rate",
        type=int,
        default=5,
        help="Messages per second",
    )

    args = parser.parse_args()

    producer = PrescriptionKafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
    )

    producer.stream(rate_per_second=args.rate)


if __name__ == "__main__":
    main()
