import numpy as np
import torch

from alphapose.models import builder
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.config import update_config

class DetectionLoader():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self._input_size = self.cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = self.cfg.DATA_PRESET.SIGMA

        if self.cfg.DATA_PRESET.TYPE == 'simple':
            pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)
        elif self.cfg.DATA_PRESET.TYPE == 'simple_smpl':
            # TODO: new features
            from easydict import EasyDict as edict
            dummpy_set = edict({
                'joint_pairs_17': None,
                'joint_pairs_24': None,
                'joint_pairs_29': None,
                'bbox_3d_shape': (2.2, 2.2, 2.2)
            })
            self.transformation = SimpleTransform3DSMPL(
                dummpy_set, scale_factor=self.cfg.DATASET.SCALE_FACTOR,
                color_factor=self.cfg.DATASET.COLOR_FACTOR,
                occlusion=self.cfg.DATASET.OCCLUSION,
                input_size=self.cfg.MODEL.IMAGE_SIZE,
                output_size=self.cfg.MODEL.HEATMAP_SIZE,
                depth_dim=self.cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2.2, 2.2),
                rot=self.cfg.DATASET.ROT_FACTOR, sigma=self.cfg.MODEL.EXTRA.SIGMA,
                train=False, add_dpg=False,
                loss_type=self.cfg.LOSS['TYPE'])


    def image_postprocess(self, img, boxes):
        cropped_boxes = torch.zeros(len(boxes), 4)
        inps = torch.zeros(len(boxes), 3, *self._input_size)
        with torch.no_grad():
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            # inps, cropped_boxes = self.transformation.align_transform(orig_img, boxes)

        return inps, cropped_boxes

class JointFinder:
    def __init__(self, cfg, min_box_area):
        self.cfg = cfg
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.min_box_area = min_box_area

        self.use_heatmap_loss = (self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss')
        self.labels = halpe_136_fullbody_points()

    def hm_to_joints(self, boxes, scores, ids, hm_data, cropped_boxes):
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        # location prediction (n, kp, 2) | score prediction (n, kp, 1)
        assert hm_data.dim() == 4

        face_hand_num = 110
        if hm_data.size()[1] == 136:
            eval_joints = [*range(0,136)]
        elif hm_data.size()[1] == 26:
            eval_joints = [*range(0,26)]
        elif hm_data.size()[1] == 133:
            eval_joints = [*range(0,133)]
        elif hm_data.size()[1] == 68:
            face_hand_num = 42
            eval_joints = [*range(0,68)]
        elif hm_data.size()[1] == 21:
            eval_joints = [*range(0,21)]
        else:
            eval_joints = [*range(0,17)]
        pose_coords = []
        pose_scores = []
        for i in range(hm_data.shape[0]):
            bbox = cropped_boxes[i].tolist()
            if isinstance(self.heatmap_to_coord, list):
                pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                    hm_data[i][eval_joints[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                    hm_data[i][eval_joints[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)
        boxes, scores, ids, preds_img, preds_scores, pick_ids = \
            pose_nms(boxes, scores, ids, preds_img, preds_scores, self.min_box_area, use_heatmap_loss=self.use_heatmap_loss)

        result = []
        for k in range(len(scores)):
            result.append(
                {
                    'keypoints':preds_img[k],
                    'kp_score':preds_scores[k],
                    'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                    'idx':ids[k],
                    'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                }
            )

        return result
    
    def keypoint_label(self, index: int) -> str:
        return self.labels[index]
    

class Pose:
    def __init__(self, cfg_file, checkpoint, batch_size, min_box_area, device):
        cfg = update_config(cfg_file)
        self.device = device
        self.batch_size = batch_size
        self.min_box_area = min_box_area
        
        # Load pose model
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET) # type: ignore
        print('Loading pose model from %s...' % (checkpoint,))
        self.pose_model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.pose_model.to(self.device)
        self.pose_model.eval()

        self.det_loader = DetectionLoader(cfg, device)
        self.joints = JointFinder(cfg, min_box_area)

    def infer(self, img, boxes, scores):
        inps, cropped_boxes = self.det_loader.image_postprocess(img, boxes)
    
        # Pose Estimation
        inps = inps.to(self.device)
        n_poses = inps.size(0)
        leftover = 0
        if n_poses % self.batch_size:
            leftover = 1
        pose_batches = n_poses // self.batch_size + leftover
        hm = []
        with torch.no_grad():
            for j in range(pose_batches):
                inps_j = inps[j * self.batch_size:min((j + 1) * self.batch_size, n_poses)]
                hm_j = self.pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)

        hm = hm.cpu()
        result = self.joints.hm_to_joints(boxes, scores, torch.zeros(len(boxes)), hm, cropped_boxes)

        return result

    def keypoint_label(self, index: int) -> str:
        return self.joints.keypoint_label(index)

def halpe_136_fullbody_points() -> list[str]:
    # 26 body keypoints
    body = ["Nose",
        "LEye",
        "REye",
        "LEar",
        "REar",
        "LShoulder",
        "RShoulder",
        "LElbow",
        "RElbow",
        "LWrist",
        "RWrist",
        "LHip",
        "RHip",
        "LKnee",
        "Rknee",
        "LAnkle",
        "RAnkle",
        "Head",
        "Neck",
        "Hip",
        "LBigToe",
        "RBigToe",
        "LSmallToe",
        "RSmallToe",
        "LHeel",
        "RHeel"]
    # face {26-93, 68 Face Keypoints}
    face = [f"face_{i}" for i in range(26,94)]
    # left hand {94-114, 21 Left Hand Keypoints}
    LHand = [f"LHand_{i}" for i in range(94,115)]
    # right hand {115-135, 21 Right Hand Keypoints}
    RHand = [f"RHand_{i}" for i in range(115,136)]

    return body + face + LHand + RHand

    