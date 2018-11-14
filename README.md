# Deeplens Sensor Identifier
AWS Deeplens Project created this past summer for @missiondata by @pcodes and @seamusmalley. The project can identify a variety of sensors that Mission Data works with.

## Machine Learning Model
The machine learning model is a retrained Inception V3 model, which was created with the Tenorflow framework. Tensorflow was the chosen framework due to its simplicity with retraining models to recognize new images.

## Deeplens Program
The Deeplens program (`sensor_identifier.py`) strays from the AWS documentation, and does not use the Intel Model Optimizer to generate a standardized model. We encountered incompatibility errors with Tensorflow and the Model Optimizer library, so we instead used the Tensorflow library directly to process images. While this resulted in large degree of lag, we were able to correctly process incoming images from the Deepelens cam and overlay what type of sensor was in the camera frame.
