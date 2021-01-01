BUCKET='cv_final'
PACKAGES_DIR='cv_final_package'
PACKAGE_NAME='face_detector_shu244-0.1.0'

gsutil cp ../dist/Face-Detector-shu244-0.1.0.tar.gz gs://$BUCKET/$PACKAGES_DIR/$PACKAGE_NAME.tar.gz
