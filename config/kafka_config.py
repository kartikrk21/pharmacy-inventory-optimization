"""
Kafka-specific configuration and utilities
"""
import os
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaConfig:
    """Kafka configuration and topic management"""
    
    # Kafka Connection
    BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29093')
    
    # Topics
    TOPIC_PRESCRIPTIONS = 'prescriptions'
    TOPIC_INVENTORY = 'inventory_updates'
    TOPIC_ORDERS = 'orders'
    TOPIC_ALERTS = 'alerts'
    TOPIC_METRICS = 'metrics'
    
    # Topic Configuration
    NUM_PARTITIONS = 3
    REPLICATION_FACTOR = 1
    
    # Producer Configuration
    PRODUCER_CONFIG = {
        'bootstrap_servers': BOOTSTRAP_SERVERS,
        'acks': 'all',
        'retries': 3,
        'max_in_flight_requests_per_connection': 5,
        'compression_type': 'gzip',
        'batch_size': 16384,
        'linger_ms': 10,
        'buffer_memory': 33554432,
        'request_timeout_ms': 30000,
        'metadata_max_age_ms': 300000
    }
    
    # Consumer Configuration
    CONSUMER_CONFIG = {
        'bootstrap_servers': BOOTSTRAP_SERVERS,
        'auto_offset_reset': 'latest',
        'enable_auto_commit': True,
        'auto_commit_interval_ms': 5000,
        'session_timeout_ms': 30000,
        'max_poll_records': 500,
        'max_poll_interval_ms': 300000,
        'fetch_min_bytes': 1,
        'fetch_max_wait_ms': 500
    }
    
    @staticmethod
    def create_topics():
        """Create all required Kafka topics"""
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=KafkaConfig.BOOTSTRAP_SERVERS,
                client_id='admin_client'
            )
            
            topics = [
                NewTopic(
                    name=KafkaConfig.TOPIC_PRESCRIPTIONS,
                    num_partitions=KafkaConfig.NUM_PARTITIONS,
                    replication_factor=KafkaConfig.REPLICATION_FACTOR
                ),
                NewTopic(
                    name=KafkaConfig.TOPIC_INVENTORY,
                    num_partitions=KafkaConfig.NUM_PARTITIONS,
                    replication_factor=KafkaConfig.REPLICATION_FACTOR
                ),
                NewTopic(
                    name=KafkaConfig.TOPIC_ORDERS,
                    num_partitions=KafkaConfig.NUM_PARTITIONS,
                    replication_factor=KafkaConfig.REPLICATION_FACTOR
                ),
                NewTopic(
                    name=KafkaConfig.TOPIC_ALERTS,
                    num_partitions=2,
                    replication_factor=KafkaConfig.REPLICATION_FACTOR
                ),
                NewTopic(
                    name=KafkaConfig.TOPIC_METRICS,
                    num_partitions=2,
                    replication_factor=KafkaConfig.REPLICATION_FACTOR
                )
            ]
            
            admin_client.create_topics(new_topics=topics, validate_only=False)
            logger.info("Kafka topics created successfully")
            
        except TopicAlreadyExistsError:
            logger.info("Topics already exist")
        except Exception as e:
            logger.error(f"Error creating topics: {e}")
        finally:
            admin_client.close()
    
    @staticmethod
    def delete_topics():
        """Delete all topics (for testing)"""
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=KafkaConfig.BOOTSTRAP_SERVERS,
                client_id='admin_client'
            )
            
            topics = [
                KafkaConfig.TOPIC_PRESCRIPTIONS,
                KafkaConfig.TOPIC_INVENTORY,
                KafkaConfig.TOPIC_ORDERS,
                KafkaConfig.TOPIC_ALERTS,
                KafkaConfig.TOPIC_METRICS
            ]
            
            admin_client.delete_topics(topics=topics)
            logger.info("Kafka topics deleted successfully")
            
        except Exception as e:
            logger.error(f"Error deleting topics: {e}")
        finally:
            admin_client.close()
    
    @staticmethod
    def list_topics():
        """List all topics"""
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=KafkaConfig.BOOTSTRAP_SERVERS,
                client_id='admin_client'
            )
            
            topics = admin_client.list_topics()
            logger.info(f"Available topics: {topics}")
            return topics
            
        except Exception as e:
            logger.error(f"Error listing topics: {e}")
            return []
        finally:
            admin_client.close()

# Auto-create topics on import
if __name__ == "__main__":
    KafkaConfig.create_topics()
    KafkaConfig.list_topics()