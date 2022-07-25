import logging

from dagster import resource
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk


class ES:
    def __init__(self,
        index: str,
        endpoint_url: str = "localhost:9200",
        use_ssl: bool = False,
        verify_certs: bool = False,
        chunk_size: int = 1000,
        thread_count: int = 16,
        queue_size: int = 16,
    ):
        self.index = index
        self.endpoint_url = endpoint_url
        self.use_ssl = use_ssl
        self.verify_certs = verify_certs
        self.chunk_size = chunk_size
        self.thread_count = thread_count
        self.queue_size = queue_size

    def __get_es_obj(self):
        print(self.use_ssl)
        return Elasticsearch(
            ['http://localhost:9200'],  # self.endpoint_url],
            # use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
        )

    def bulk_write(self, items, logger=None):
        _logger = logging.getLogger("ES") if logger is None else logger
        pb = parallel_bulk(
            self.__get_es_obj(),
            items,
            chunk_size=self.chunk_size,
            thread_count=self.thread_count,
            queue_size=self.queue_size
        )
        errors = []
        for success, info in pb:
            if not success:
                _logger.exception(f"Failed to write document to Elasticsearch: {info}")
                errors.append(info)
        return errors


@resource(
    config_schema={
        "endpoint_url": str,
        "use_ssl": bool,
        "verify_certs": bool,
        "chunk_size": int,
        "thread_count": int,
        "queue_size": int,
    }
)
def elasticsearch(context):
    return ES(
        context.resource_config["endpoint_url"],
        context.resource_config["use_ssl"],
        context.resource_config["verify_certs"],
        context.resource_config["chunk_size"],
        context.resource_config["thread_count"],
        context.resource_config["queue_size"],
    )
