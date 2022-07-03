from dagster import resource

from s3fs import S3FileSystem
from fsspec.implementations.local import LocalFileSystem


@resource
def local_file_system():
    return LocalFileSystem()


@resource
def s3_file_system():
    return S3FileSystem()

