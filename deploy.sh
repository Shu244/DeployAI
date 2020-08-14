#!/bin/bash

MODEL_NAME='torch_image_caption'
RUNTIME_VERSION='1.13'
REGION='us-central1'
VERSION_NAME='image_caption_1'

gcloud ai-platform models create ${MODEL_NAME} --regions ${REGION}

# Only need to run once
gcloud components install beta

gcloud beta ai-platform versions create ${VERSION_NAME} \
 --model ${MODEL_NAME} \
 --runtime-version ${RUNTIME_VERSION} \
 --python-version 3.7 \
 --origin gs://integral-accord-270805-deploy-ai/image_caption \
 --package-uris gs://integral-accord-270805-deploy-ai/image_caption-0.1.tar.gz \
 --prediction-class model_prediction.CustomModelPrediction
