"""
Anomify scoring Function.

Trigger : Event Hub  "windowed-batches"   (filled by the Stream Analytics job)
Outputs : Event Hub  "anomaly-alerts"     (only written to when is_anomaly == True)
          Event Hub  "scores"             (written for EVERY batch - dashboard/monitoring feed)

WHY an Event Hub trigger instead of HTTP:
Stream Analytics can only write to a sink it understands (Event Hub, Blob,
SQL, Cosmos DB, Power BI, ...). It cannot call an HTTP endpoint directly.
So Stream Analytics -> Event Hub -> Function is the standard "glue" pattern.

The model + preprocessor + scaler are loaded ONCE at cold start (module
import time), not per-invocation - reloading a Keras model on every event
would make this unusably slow and is unnecessary since the container/worker
process is reused across invocations.
"""
import json
import logging
import os
import sys
from pathlib import Path

import azure.functions as func
import pandas as pd

# The project code (pipelines/inference.py, src/*) is copied into the image
# at /home/site/wwwroot/project - see Dockerfile.
sys.path.insert(0, str(Path(__file__).parent / "project"))
from pipelines.inference import AnomifyLiveDetector  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anomify")

logger.info("Cold start: loading model, scaler and preprocessing pipeline...")
detector = AnomifyLiveDetector()
logger.info("Cold start complete - detector ready.")

app = func.FunctionApp()


@app.function_name(name="ScoreWindow")
@app.event_hub_message_trigger(
    arg_name="event",
    event_hub_name="windowed-batches",
    connection="EVENTHUB_CONNECTION",
)
@app.event_hub_output(
    arg_name="scoreOut",
    event_hub_name="scores",
    connection="EVENTHUB_CONNECTION",
)
@app.event_hub_output(
    arg_name="alertOut",
    event_hub_name="anomaly-alerts",
    connection="EVENTHUB_CONNECTION",
)
def ScoreWindow(event: func.EventHubEvent, scoreOut: func.Out[str], alertOut: func.Out[str]):
    payload = json.loads(event.get_body().decode("utf-8"))

    # Shape produced by windowing.asaql: {"deviceId": ..., "windowEnd": ..., "readings": [ {...}, ... ]}
    device_id = payload.get("deviceId", "unknown")
    window_end = payload.get("windowEnd")
    readings = payload["readings"]

    raw_df = pd.DataFrame(readings)

    try:
        result = detector.score_batch(raw_df)
    except Exception as exc:
        logger.error(f"Scoring failed for device={device_id}: {exc}")
        return

    result["deviceId"] = device_id
    result["windowEnd"] = window_end

    logger.info(f"Scored window for {device_id}: mse={result['mse']:.6f} anomaly={result['is_anomaly']}")

    # Every score goes to the monitoring feed (e.g. Power BI streaming dataset).
    scoreOut.set(json.dumps(result))

    # Only anomalies go to the alerts feed (e.g. feeding a Logic App / email / Teams webhook).
    if result["is_anomaly"]:
        alertOut.set(json.dumps(result))
