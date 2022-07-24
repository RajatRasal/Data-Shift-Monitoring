from dagster import resource, Noneable

from s3fs import S3FileSystem
from fsspec.implementations.local import LocalFileSystem


@resource
def local_file_system():
    return LocalFileSystem()


@resource(config_schema={"endpoint_url": Noneable(str)})
def s3_file_system(context):
    # TODO: Put secret key, access key and endpoint url into one file.
    return S3FileSystem(
        client_kwargs={
            "endpoint_url": context.resource_config["endpoint_url"],
        }
    )
