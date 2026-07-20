from ice_data_py import cds
from datetime import datetime
from typing import Any

def load_data(
        measurement_system: str,
        measurement_name: str,
        measurement_parameter: str,
        start_time: str | datetime,
        end_time: str | datetime
) -> Any:
    """
    Fetch measurement data from the data store.

    Args:
        measurement_system: The measurement system identifier
        measurement_name: The name of the measurement
        measurement_parameter: The parameter to measure
        start_time: Start time for data retrieval (ISO format string or datetime object)
        end_time: End time for data retrieval (ISO format string or datetime object)

    Returns:
        Fetched measurement data

    Raises:
        ValueError: If start_time is after end_time
    """
    if start_time > end_time:
        raise ValueError("start_time must be before end_time")

    query_spec = cds.QuerySpec(measurement_system, measurement_name, measurement_parameter)
    return cds.fetch_all([query_spec], start_time, end_time)
