#!/bin/bash

MODEL_NAME='face_detector'
RUNTIME_VERSION='2.2'
REGION='us-central1'
VERSION_NAME='face_detector_1'

gcloud ai-platform models create ${MODEL_NAME} --regions ${REGION}

# Only need to run once
gcloud components install beta

gcloud beta ai-platform versions create ${VERSION_NAME} \
 --model ${MODEL_NAME} \
 --runtime-version ${RUNTIME_VERSION} \
 --python-version 3.7 \
 --origin gs://cv_final/model_dir \
 --package-uris gs://cv_final/cv_final_package/face_detector_shu244-0.1.0.tar.gz,gs://sand-box-shu/torchvision-0.7.0a0+78ed10c-cp38-cp38-linux_x86_64.whl \
 --prediction-class CustomModelPrediction.CustomModelPrediction \
 --machine-type mls1-c4-m4 # Prevents out of memory error

