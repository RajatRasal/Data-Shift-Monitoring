import logging

from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from dagster import resource


class ES:
    def __init__(self, index):
        self.index = index

    def bulk_write(self, items, logger=None):
        _logger = logging.getLogger("ES") if logger is None else logger
        es = Elasticsearch()
        pb = parallel_bulk(es, items, chunk_size=10000, thread_count=16, queue_size=16)
        errors = []
        for success, info in pb:
            if not success:
                _logger.exception(f"Failed to write document to Elasticsearch: {info}")
                errors.append(info)
        return errors


@resource(config_schema={"mapping": str})
def elasticsearch(context):
    return ES()