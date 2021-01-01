import io
import os
import numpy as np

from PIL import Image
from google.cloud import storage


class CustomModelPrediction(object):
    def __init__(self, model):
        # Stores artifacts for prediction. Only initialized via `from_path`.
        self._model = model
        self.read_storage_client = storage.Client()

    def predict(self, instances, **kwargs):
        """
        Performs custom prediction.

        Instances are the decoded values from the request. They have already
        been deserialized from JSON.

        Args:
            instances: A list of prediction input instances.
            **kwargs: A dictionary of keyword args provided as additional
                fields on the predict request body.

        Returns:
            A list of outputs containing the prediction results. This list must
            be JSON serializable.
        """

        results = []
        for bucket_name, blob_name in instances:
            image = self.get_image_from_bucket(bucket_name, blob_name)
            result = self._model(image)
            results.append(result)
        return results

    def get_image_from_bucket(self, bucket_name, blob_name):
        '''
        Pulls image from Google Cloud Storage
        :param bucket_name: Bucket name the image is stored in
        :param blob_name: The name of the blob representing the image
        :return: Numpy array of the image in RGB
        '''
        bucket = self.read_storage_client.get_bucket(bucket_name)
        blob = bucket.get_blob(blob_name)
        image_bytes = blob.download_as_bytes()
        image = np.array(Image.open(io.BytesIO(image_bytes)))
        return image

    @classmethod
    def from_path(cls, model_dir):
        """
        Creates an instance of CustomModelPrediction using the given path.

        This loads artifacts that have been copied from your model directory in
        Cloud Storage. CustomModelPrediction uses them during prediction.

        Args:
            model_dir

        Returns:
            An instance of `MyPredictor`.
        """

        from face_detector.interface import FaceDetectorInterface
        param_path = os.path.join(model_dir, "yolov3_ckpt_best.pth")
        model_arch = os.path.join(model_dir, "yolov3-custom.cfg")
        class_path = os.path.join(model_dir, "classes.names")
        interface = FaceDetectorInterface.create_old(param_path, model_arch, class_path)
        return cls(interface)

