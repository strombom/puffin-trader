
from datetime import datetime, timezone


def string_to_timestamp(date, fmt='%Y-%m-%d %H:%M:%S.%f'):
    return datetime.timestamp(datetime.strptime(date, fmt))


def string_to_datetime(date, fmt='%Y-%m-%d %H:%M:%S.%f'):
    return datetime.fromtimestamp(string_to_timestamp(date, fmt), tz=timezone.utc)


def timestamp_to_string(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
