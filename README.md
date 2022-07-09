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

### Run
- `docker pull prom/pushgateway`
- `docker run -d -p 9091:9091 prom/pushgateway`
- `dagit -f dag.py`
