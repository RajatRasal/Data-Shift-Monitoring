from dagster import resource


class ES:
    def __init__(self):
        pass

    def log(self, item):
        pass


@resource
def elasticsearch():
    return ES()