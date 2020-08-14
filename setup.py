from setuptools import setup

REQUIRED_PACKAGES = ['torch', 'imageio', 'opencv']

setup(
 name="image_caption_ai",
 version="0.1",
 scripts=["models.py", "model_prediction.py", "caption_custom.py"],
 install_requires=REQUIRED_PACKAGES
)
