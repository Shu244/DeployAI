#!/bin/bash

BUCKET='cv_final'
REGION='us-central1'
MODEL_DIR='model_dir'

PARAM_PTH='../face_detector/checkpoints/yolov3_ckpt_2.pth'
ARCH_PTH='../face_detector/config/yolov3-custom.cfg'
CLASS_NAMES='../face_detector/WIDER/data/classes.names'

gsutil mb -l $REGION gs://$BUCKET

gsutil cp $PARAM_PTH gs://$BUCKET/$MODEL_DIR/
gsutil cp $ARCH_PTH gs://$BUCKET/$MODEL_DIR/
gsutil cp $CLASS_NAMES gs://$BUCKET/$MODEL_DIR/
