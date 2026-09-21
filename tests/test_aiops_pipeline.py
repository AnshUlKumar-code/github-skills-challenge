import io
import json
import runpy
from contextlib import redirect_stdout
from pathlib import Path

from src.anomaly_detector import AnomalyDetector
from src.aiops_pipeline import run_pipeline
from src.event_consumer import EventConsumer
from src.event_producer import EventProducer
from src.event_topic import EventTopic


def test_normal_record_is_not_anomaly():
    detector = AnomalyDetector()

    record = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "payment-service",
        "response_time_ms": 120,
        "cpu_percent": 42,
        "memory_percent": 51,
        "log_level": "INFO",
        "message": "Payment request processed successfully"
    }

    assert detector.detect(record) is None


def test_anomalous_record_is_detected():
    detector = AnomalyDetector()

    record = {
        "timestamp": "2026-09-20T10:05:00",
        "service": "payment-service",
        "response_time_ms": 610,
        "cpu_percent": 75,
        "memory_percent": 70,
        "log_level": "ERROR",
        "message": "Payment service timeout"
    }

    event = detector.detect(record)

    assert event is not None
    assert event["type"] == "ANOMALY"


def test_producer_publishes_event():
    topic = EventTopic("anomaly-events")
    producer = EventProducer(topic)

    event = {
        "type": "ANOMALY",
        "service": "payment-service"
    }

    assert producer.publish(event)
    assert len(topic.get_messages()) == 1


def test_consumer_receives_event():
    topic = EventTopic("anomaly-events")
    producer = EventProducer(topic)
    consumer = EventConsumer(topic)

    event = {
        "type": "ANOMALY",
        "service": "payment-service"
    }

    producer.publish(event)

    messages = consumer.consume()

    assert len(messages) == 1


def test_producer_rejects_empty_event():
    topic = EventTopic("anomaly-events")
    producer = EventProducer(topic)

    assert producer.publish({}) is False
    assert topic.get_messages() == []


def test_topic_clear_removes_messages():
    topic = EventTopic("anomaly-events")
    topic.publish({"type": "ANOMALY"})

    topic.clear()

    assert topic.get_messages() == []


def test_detector_catches_cpu_and_memory_anomalies():
    detector = AnomalyDetector()

    record = {
        "timestamp": "2026-09-20T10:06:00",
        "service": "billing-service",
        "response_time_ms": 400,
        "cpu_percent": 81,
        "memory_percent": 90,
        "log_level": "INFO",
        "message": "CPU spike while processing invoice"
    }

    event = detector.detect(record)

    assert event is not None
    assert event["type"] == "ANOMALY"
    assert "High CPU utilization" in event["reasons"]
    assert "High memory utilization" in event["reasons"]


def test_run_pipeline_returns_detected_anomalies():
    result = run_pipeline("data/service_data.json")

    assert result["records_processed"] == 10
    assert len(result["anomalies_detected"]) == 2
    assert len(result["events_consumed"]) == 2
    assert all(event["type"] == "ANOMALY" for event in result["events_consumed"])


def test_run_pipeline_on_clean_data_returns_empty_lists():
    data = [{
        "timestamp": "2026-09-20T10:00:00",
        "service": "checkout-service",
        "response_time_ms": 120,
        "cpu_percent": 42,
        "memory_percent": 51,
        "log_level": "INFO",
        "message": "Healthy request"
    }]

    temp_path = Path("/tmp/clean_service_data.json")
    temp_path.write_text(json.dumps(data), encoding="utf-8")

    try:
        result = run_pipeline(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)

    assert result["records_processed"] == 1
    assert result["anomalies_detected"] == []
    assert result["events_consumed"] == []


def test_aiops_pipeline_main_block_runs():
    output = io.StringIO()

    with redirect_stdout(output):
        runpy.run_module("src.aiops_pipeline", run_name="__main__")

    stdout = output.getvalue()
    assert "AIOps Pipeline Result" in stdout
    assert "Records processed: 10" in stdout
    assert "Anomalies detected: 2" in stdout
    assert "Events consumed: 2" in stdout