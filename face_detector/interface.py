from __future__ import division

import cv2
import os
import urllib
import matplotlib.pyplot as plt

from face_detector.models import *
from face_detector.utils.utils import *
from face_detector.utils.datasets import *
from torch.autograd import Variable


class FaceDetectorInterface:
    # Class is necessary to prevent having to load the model everytime run method is called
    def __init__(self, configs):
        self.cfigs = configs
        self.model = self.gen_model()
        self.classes = load_classes(self.cfigs['class_path'])

    def gen_model(self):
        model = Darknet(self.cfigs['model_arch'], img_size=self.cfigs['img_size'])
        model = model.to(self.cfigs['device'])
        model.load_state_dict(torch.load(self.cfigs['param_path'], map_location=self.cfigs['device']))
        model.eval()
        return model

    def run(self, image=None, image_path=None):
        '''
        Return a list of maps. Each map represents a face and contains information
        about it (corner coordinates and more).

        :param image: A ndarray with dimension  H x W x C
        :return: List of maps
        '''

        if image is None and image_path is None:
            raise ValueError
        if image_path:
            image = FaceDetectorInterface.read_image(image_path)


        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


        H, W, C = image.shape
        img = transforms.ToTensor()(image)
        img, _ = pad_to_square(img, 0)
        img = resize(img, self.cfigs['img_size'])
        img = Variable(img.type(Tensor)[:3, :, :])
        img = img.unsqueeze(0)

        with torch.no_grad():
            batch_detections = self.model(img)
            batch_detections = non_max_suppression(batch_detections, self.cfigs['conf_thres'], self.cfigs['nms_thres'])

            detections = batch_detections[0]
            detections = rescale_boxes(detections, self.cfigs['img_size'], (H, W))

            results = []
            for detection in detections:
                x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
                label = self.classes[int(cls_pred)]
                result = {
                    'x1': int(round(x1.item())),
                    'y1': int(round(y1.item())),
                    'x2': int(round(x2.item())),
                    'y2': int(round(y2.item())),
                    'label': label,
                    'object_confidence': conf.item(),
                    'class_confidence': cls_conf.item(),
                    'class_prediction': cls_pred.item()
                }
                results.append(result)
            return results

    @staticmethod
    def read_image(img_path):
        '''
        Returns an ndarray of dimensions H x W x C
        :param img_path: Path for image
        :return: Returns image including alpha channel
        '''

        return np.array(Image.open(img_path))

    @staticmethod
    def plot_faces(image, results):
        for result in results:
            x1 = result['x1']
            x2 = result['x2']
            y1 = result['y1']
            y2 = result['y2']
            start = x1, y1
            end = x2, y2
            face = cv2.getRectSubPix(image, (x2 - x1, y2 - y1),
            ((x2 + x1) / 2, (y2 + y1) / 2))
            
            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

            color = (0, 255, 0)
            cv2.rectangle(image, start, end, color, 2)

        plt.figure()
        plt.imshow(image)
        plt.show()

    @staticmethod
    def create_old(param_path='face_detector/checkpoints/yolov3_ckpt_13.pth',
                   model_arch='face_detector/config/yolov3-custom.cfg',
                   class_path='face_detector/config/classes.names'):

        configs = FaceDetectorInterface.get_configs(param_path, model_arch, class_path)
        return FaceDetectorInterface(configs)

    @staticmethod
    def create(folder):

        if not os.path.isdir(folder):
            print('Creating new folder to download params into')
            os.makedirs(folder)

        param_path = os.path.join(folder, 'yolov3_ckpt_best.pth')
        model_arch = os.path.join(folder, 'yolov3-custom.cfg')
        class_path = os.path.join(folder, 'classes.names')

        if not os.path.isfile(param_path) or not os.access(param_path, os.R_OK):
            print('Downloading parameters to path:', os.path.abspath(param_path))
            url = 'https://storage.googleapis.com/cv_final/model_dir/yolov3_ckpt_best.pth'
            urllib.request.urlretrieve(url, param_path)
            print('Finished')
        if not os.path.isfile(model_arch) or not os.access(model_arch, os.R_OK):
            print('Downloading model architecture to path:', os.path.abspath(model_arch))
            url = 'https://storage.googleapis.com/cv_final/model_dir/yolov3-custom.cfg'
            urllib.request.urlretrieve(url, model_arch)
            print('Finished')
        if not os.path.isfile(class_path) or not os.access(class_path, os.R_OK):
            print('Downloading class names to path:', os.path.abspath(class_path))
            url = 'https://storage.googleapis.com/cv_final/model_dir/classes.names'
            urllib.request.urlretrieve(url, class_path)
            print('Finished')

        configs = FaceDetectorInterface.get_configs(param_path, model_arch, class_path)
        return FaceDetectorInterface(configs)

    @staticmethod
    def get_configs(param_path, model_arch, class_path):
        return {
            'param_path': param_path,
            'model_arch': model_arch,
            'class_path': class_path,
            'conf_thres': 0.8,
            'nms_thres': 0.4,
            'batch_size': 1,
            'n_cpu': 0,
            'img_size': 416,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }


if __name__=='__main__':
    img_path = 'example.jpeg'
    interface = FaceDetectorInterface.create(folder='checkpoints')
    res = interface.run(image_path=img_path)
    print(res)
