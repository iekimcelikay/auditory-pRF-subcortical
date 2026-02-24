"""Timestamp generation utilities."""

from datetime import datetime


# Common format strings
class TimestampFormats:
    """Common timestamp format strings."""
    COMPACT = "%Y%m%d_%H%M%S"
    READABLE = "%Y-%m-%d_%H:%M:%S"
    DATE_ONLY = "%Y-%m-%d"
    TIME_ONLY = "%H-%M-%S"


def generate_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generate a timestamp string based on the current time.

    Args:
        format_string: Format string for datetime.strftime
                      Default: YYYYMMDD_HHMMSS

    Returns:
        Formatted timestamp string

    Examples:
        >>> generate_timestamp()
        '20250122_143022'
        >>> generate_timestamp(TimestampFormats.READABLE)
        '2025-01-22_14:30:22'
    """
    return datetime.now().strftime(format_string)


def generate_unix_epoch() -> int:
    """
    Generate the current time as a Unix epoch timestamp.

    Returns:
        Unix epoch timestamp (integer seconds since 1970-01-01)

    Examples:
        >>> generate_unix_epoch()
        1737557422
    """
    return int(datetime.now().timestamp())


def get_current_datetime() -> datetime:
    """
    Get the current datetime object.

    Returns:
        Current datetime object

    Examples:
        >>> dt = get_current_datetime()
        >>> dt.year
        2025
    """
    return datetime.now()