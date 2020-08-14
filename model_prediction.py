import os


class CustomModelPrediction(object):
    def __init__(self, model):
        # Stores artifacts for prediction. Only initialized via `from_path`.
        self._model = model

    def predict(self, instance, **kwargs):
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
        caption_str = self._model(instance)
        return caption_str

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

        from caption_custom import Caption

        model_path = os.path.join(model_dir, 'model.tar')
        embedding_path = os.path.join(model_dir, 'embeddings.json')

        caption = Caption(model_path, embedding_path)

        return cls(caption)