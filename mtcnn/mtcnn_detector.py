from mtcnn.MtcnnDetector import MtcnnDetector
from mtcnn.detector import Detector
from mtcnn.fcn_detector import FcnDetector
from mtcnn.mtcnn_model import P_Net, R_Net, O_Net


def get_mtcnn_detector(prefix=['../mtcnn/MTCNN_model/PNet_landmark/PNet', '../mtcnn/MTCNN_model/RNet_landmark/RNet', '../mtcnn/MTCNN_model/ONet_landmark/ONet']):
    thresh = [0.9, 0.6, 0.7]
    min_face_size = 24
    stride = 2
    slide_window = False
    detectors = [None, None, None]
    epoch = [18, 14, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    RNet = Detector(R_Net, 24, 1, model_path[1])
    detectors[1] = RNet
    ONet = Detector(O_Net, 48, 1, model_path[2])
    detectors[2] = ONet
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)
    return mtcnn_detector
