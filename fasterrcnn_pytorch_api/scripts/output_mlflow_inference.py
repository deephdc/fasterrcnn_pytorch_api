ubuntu@mlflow:/home/fasterrcnn/# python fasterrcnn_pytorch_api/scripts/mlflow_inference.py 
**output**

Password: 
Device: cpu
Available experiments:
nyc-taxi-exp-prefect
fasterrcnn_exp
deepaas_full-testing
deepaas_full-server
random-forest-best-models
random-forest-hyperopt
mlops_zoomcamp
customer-sentiment-analysis
wind_power_forecast
wine_#1
diabetes_#ex5
Default
Enter the name of the experiment: fasterrcnn_exp
Runs and Models in experiment 'fasterrcnn_exp':
Run ID: 9671cadd75bb42f4b0fe460ffe12f787
Run Name: 20230710_155158
No models found for this run.
--------------------------------------------------
Run ID: be2700f5fc364da78d66378f6e474edd
Run Name: 20230710_151753
Model Version: 1
Model Name: fasterrcnn
Model Stage: Production
--------------------------------------------------
Run ID: 43e8d18432ca4beea3bdfc14f7960d7c
Run Name: 20230705_182531
Model Version: 1
Model Name: test_fasterrcnn
Model Stage: None
--------------------------------------------------

Select the Model Name (in production): fasterrcnn
2023/07/17 11:51:43 WARNING mlflow.pytorch: Stored model version '2.0.1+cu117' does not match installed PyTorch version '1.13.0'
Model loaded successfully.
num_class 6
Test instances: 1
Image 1 done...
--------------------------------------------------
TEST PREDICTIONS COMPLETE
Average FPS: 2.026