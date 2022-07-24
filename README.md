# Data Shift Monitoring

A demo of how to monitor data shift on a vision task in an ML batch processing pipeline. 

There are 4 main stages to data shift monitoring:
1. Data Accumulation
1. Data Modelling
1. Distribution Comparison - `https://scikit-multiflow.readthedocs.io/en/stable/api/api.html#module-skmultiflow.drift_detection`
1. Hypothesis Test - `from alibi_detect.cd import KSDrift`

### Papers
- Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift - https://arxiv.org/pdf/1810.11953.pdf
- Mandoline: Model Evaluation under Distribution Shift - https://arxiv.org/pdf/2107.00643.pdf, https://github.com/HazyResearch/mandoline
- An overview of unsupervised drift detection methods - https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1381
- https://www.microsoft.com/en-us/research/uploads/prod/2019/03/amershi-icse-2019_Software_Engineering_for_Machine_Learning.pdf
- https://storage.googleapis.com/pub-tools-public-publication-data/pdf/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf

### Articles
- https://madewithml.com/courses/mlops/monitoring/
- https://towardsdatascience.com/drift-metrics-how-to-select-the-right-metric-to-analyze-drift-24da63e497e
- https://www.jeremyjordan.me/ml-monitoring/

### Frameworks
- https://evidentlyai.com/
- https://torchdrift.org/notebooks/drift_detection_on_images.html
- https://whylogs.readthedocs.io/en/latest/
- https://www.nannyml.com/

### Architecture
- Dagster - orchestration
- Feast - feature store
- MLFlow - model tracking and storage
- Elasticsearch - search index
- Postgres - state management https://qbox.io/blog/maximize-guide-elasticsearch-indexing-performance-part-1?utm_source=qbox.io&utm_medium=article&utm_campaign=maximize-guide-elasticsearch-indexing-performance-part-2, https://qbox.io/blog/maximize-guide-elasticsearch-indexing-performance-part-2 

### Run
#### Prometheus
- `docker pull prom/pushgateway`
- `docker run -d -p 9091:9091 prom/pushgateway`

#### Minio
- ```docker run \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ROOT_USER=123" \
  -e "MINIO_ROOT_PASSWORD=12345678" \
  -v "/Users/work/Documents/Data-Shift-Monitoring/.minio_data:/data" \
  quay.io/minio/minio server /data --console-address ":9001"```
- `brew install minio/stable/mc`
- `mc --insecure alias set minio http://localhost:9000 123 12345678`
- `AWS_ACCESS_KEY_ID=123 AWS_SECRET_ACCESS_KEY=12345678 aws --endpoint-url http://localhost:9000 s3 mb --ignore-existing landing-bay`
- `AWS_ACCESS_KEY_ID=123 AWS_SECRET_ACCESS_KEY=12345678 aws --endpoint-url http://localhost:9000 s3 mb --ignore-existing pdf-pages`

#### Elasticsearch
- `docker-compose -f infra/elasticsearch/docker-compose.yaml up`

#### Dagit
- `AWS_ACCESS_KEY_ID=123 AWS_SECRET_ACCESS_KEY=12345678 dagit -f dag.py`

### TODO:
FastAPI:
1. All pdfs found with processing state
1. For a pdf return image + text + model id + confidence score
1. Find pdfs that have a word in them

Postgres:
1. State for PDF discovery - path
1. State for each image - stored, ocr, es
1. Tables - PDF names -> (PDF, image, states) <- state

Helm charts OR Docker compose for all services - dagster, minio, flask app, bootstrap frontend, elasticsearch, grafana, prometheus
