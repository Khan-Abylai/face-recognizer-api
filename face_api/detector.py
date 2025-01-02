from models.retinaface import RetinaFace
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from layers.functions.prior_box import PriorBox
from utils import decode, decode_landm, py_cpu_nms
from align_face import align_img
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
def load_model(model, pretrained_path, load_to_cpu):
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class Detector(object):
    def __init__(self, weight_path, cfg_mnet):
        self.device = torch.device("cuda")
        self.cfg_mnet = cfg_mnet
        self.net = RetinaFace(cfg=cfg_mnet, phase='test')
        self.net = load_model(self.net, weight_path, load_to_cpu=False)
        self.net.eval()
        cudnn.benchmark = True
        self.net = self.net.to(self.device)

    def detect(self, img_raw, confidence_threshold, nms_threshold):
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg_mnet['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg_mnet['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:750]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:750, :]
        landms = landms[:750, :]

        dets = np.concatenate((dets, landms), axis=1)
        if len(dets):
            for b in dets:
                # text = "{:.4f}".format(b[4])
                # b = list(map(int, b))
                # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                # cx = b[0]
                # cy = b[1] + 12
                # cv2.putText(img_raw, text, (cx, cy),
                #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                #
                # # landms
                # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

                # face align
                result_landmark = np.array([b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14]])
                landmark5 = result_landmark.astype(int)
                crop_img = align_img(img_raw, landmark5)

            return crop_img
        else:
            return None
