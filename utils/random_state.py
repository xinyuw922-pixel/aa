"""JSON-compatible random-state conversion without project dependencies."""


def tuples(value):
    """Restore nested tuples required by random.setstate after JSON transport."""
    return tuple(tuples(item) for item in value) if isinstance(value, list) else value
