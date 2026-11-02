"""Leaf display formatting for optional metric values."""


def format_percentage(value: float | None) -> str:
    """Display an unavailable estimate without turning it into a numeric zero."""
    return 'N/A' if value is None else f'{value:.1%}'
