"""Utilities for calendar-date prefix evaluation."""

import datetime as dt


def parse_cutoff_dates(value, reference_date):
    """Parse comma-separated ISO dates as sorted (date, relative day) pairs."""
    if not value:
        raise ValueError("请通过 --cutoff_dates 提供至少一个 YYYY-MM-DD 日期")
    ref = dt.datetime.strptime(reference_date, "%Y-%m-%d").date()
    parsed = []
    for item in value.split(","):
        date_value = dt.datetime.strptime(item.strip(), "%Y-%m-%d").date()
        parsed.append((date_value.isoformat(), (date_value - ref).days))
    return sorted(set(parsed), key=lambda item: item[1])
