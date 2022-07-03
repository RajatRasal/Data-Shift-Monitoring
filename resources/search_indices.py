from dagster import resource


@resource
def elasticsearch():
    return None