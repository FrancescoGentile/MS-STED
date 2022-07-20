##
## Code based on CTR-GCN
##

from typing import List, Tuple
import os
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import yaml

from src.dataset.ntu.config import NTUDatasetConfig
from src.dataset.generator import DatasetGenerator
import src.utils as utils

NOISE_LENGTH_THRESHOLD = 11
NOISE_SPREAD_THRESHOLD1 = 0.8
NOISE_SPREAD_THRESHOLD2 = 0.69754

class NTUDatasetGenerator(DatasetGenerator):
    
    def __init__(self, config: NTUDatasetConfig) -> None:
        self.config = config
        log_file = os.path.join(self.config.dataset_path, 'log.txt')
        self.logger: logging.Logger = utils.init_logger(name=self.config.name, 
                                                        level=logging.INFO,
                                                        file=log_file)
        
        ignored_samples = []
        if self.config.ignored_file is not None:
            try:
                with open(self.config.ignored_file, 'r') as f:
                    ignored_samples = [f'{line.strip()}.skeleton' for line in f.readlines()]
            except Exception as e:
                if self.config.debug:
                    self.logger.exception(e)
                else:
                    self.logger.error(e)
                raise e
        else:
            self.logger.error(f'No file with ignored samples was specified.')
            raise ValueError(f'No file with ignored samples was specified.')
                
        
        self.file_list = []
        for folder in [self.config.ntu60_path, self.config.ntu120_path]:
            if folder is None:
                break
            for filename in os.listdir(folder):
                if filename not in ignored_samples:
                    self.file_list.append((folder, filename))
                    
    def _get_training_samples(self) -> list:
        training_samples = {}
        training_samples['ntu60-xsub'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
        ]
        training_samples['ntu60-xview'] = [2, 3]
        training_samples['ntu120-xsub'] = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
            38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
            80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
        ]
        training_samples['ntu120-xset'] = set(range(2, 33, 2))
        
        return training_samples[self.config.name]
        
    def _get_sequence_raw_bodies(self, path: str, filename: str) -> dict:
        """
        Get raw bodies data from a ntu skeleton file.
        
        Args:
            path (str): directory containing the file
            filename (str): name of the file

        Returns:
            dict: a dictionary with the following keys
            - name: the skeleton filename 
            - data: a dictionary that stores raw data of each body (body_id, body_data),
                where body_data is a dictionary with the following structure: 
                - interval: list of frames index where this body appears
                - joints: ndarray of shape (len(interval), 25, 3)
                - colors: ndarray of shape (len(interval), 25, 2)
            - num_frames: the number of valid frames
        """
        
        full_path = os.path.join(path, filename)
        if not os.path.isfile(full_path):
            e = ValueError(f'Skeleton file {full_path} does not exist.')
            if self.config.debug:
                self.logger.exception(e)
            else:
                self.logger.error(e)
            raise e

        with open(full_path, 'r') as fr:
            num_frames = int(fr.readline())
            frames_drop = []
            bodies_data = {}
            valid_frames = -1
            
            for frame in range(num_frames):
                num_bodies = int(fr.readline())
                
                if num_bodies == 0: 
                    frames_drop.append(frame)
                    continue
                
                valid_frames += 1
                joints = np.zeros((num_bodies, self.config.num_joints, self.config.num_coords), dtype=np.float32)
                colors = np.zeros((num_bodies, self.config.num_joints, 2), dtype=np.float32)
                
                for body in range(num_bodies):
                    body_info = fr.readline().strip('\r\n').split()
                    body_id = body_info[0]
                    num_joints = int(fr.readline())
                    
                    for joint in range(num_joints):
                        joint_info = fr.readline().strip('\r\n').split()
                        joints[body, joint, :] = np.array(joint_info[:self.config.num_coords], dtype=np.float32)
                        colors[body, joint, :] = np.array(joint_info[5:7], dtype=np.float32)
                        
                    if body_id not in bodies_data: # add a new body
                        body_data = {'joints': joints[body], 
                                     'colors': colors[body, np.newaxis], 
                                     'interval': [valid_frames]}
                    else: # update already existing body
                        body_data = bodies_data[body_id]
                        # Stack each body's data of each frame along the frame order
                        body_data['joints'] = np.vstack((body_data['joints'], joints[body]))
                        body_data['colors'] = np.vstack((body_data['colors'], colors[body, np.newaxis]))
                        pre_frame_idx = body_data['interval'][-1]
                        body_data['interval'].append(pre_frame_idx + 1)
                        
                    bodies_data[body_id] = body_data
        
        if len(bodies_data) > 1: 
            for body_data in bodies_data.values():
                body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

        return {'name': filename.split('.')[0], 
                'data': bodies_data, 
                'num_frames': num_frames - len(frames_drop)}
    
    def _get_one_actor_points(self, 
                              body_data: dict, 
                              num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joints and colors for only one actor. 
        
        Args:
            body_data (dict): a dictionary containing the body data for each frame
            num_frames (int): the number of total frames

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
            joints tensor with shape (len(num_frames), 75), where 75 = 25 * 3 (xyz coordinates),
            colors tensor with shape (len(num_frames), 50), where 50 = 25 * 2 (xy coordinates)
        """
        
        joints = np.zeros((num_frames, 25 * 3), dtype=np.float32)
        colors = np.zeros((num_frames, 1, 25, 2), dtype=np.float32)
        start, end = body_data['interval'][0], body_data['interval'][-1]
        joints[start:end + 1] = body_data['joints'].reshape(-1, 75)
        colors[start:end + 1, 0] = body_data['colors']
        
        return joints, colors
    
    def _denoise_by_length(self, bodies_data: dict) -> dict:
        """
        Denoising data based on the frame length for each bodyID.
        Filter out the bodyID which length is less or equal than the predefined threshold.
        
        Args:
            bodies_data (dict):

        Returns:
            dict: denoised bodies 
        """
        new_bodies_data = bodies_data.copy()
        for (bodyID, body_data) in new_bodies_data.items():
            length = len(body_data['interval'])
            if length <= NOISE_LENGTH_THRESHOLD:
                del bodies_data[bodyID]

        return bodies_data
    
    def _get_valid_frames_by_spread(self, points: np.ndarray) -> List[int]:
        """
        Find the valid (or reasonable) frames (index) based on the spread of X and Y.
        
        Args:
            points (np.ndarray): joints or colors

        Returns:
            List[int]: list of frames indexes
        """
        
        num_frames = points.shape[0]
        valid_frames = []
        for i in range(num_frames):
            x = points[i, :, 0]
            y = points[i, :, 1]
            if (x.max() - x.min()) <= NOISE_SPREAD_THRESHOLD1 * (y.max() - y.min()):  # 0.8
                valid_frames.append(i)
        return valid_frames
    
    def _denoise_by_spread(self, bodies_data: dict) -> dict:
        """
        Denoising data based on the spread of Y value and X value.
        Filter out the bodyID which the ratio of noisy frames is higher than the predefined threshold.
        
        Args:
            bodies_data (dict): bodies data

        Returns:
            dict: denoised bodies
        """
        
        new_bodies_data = bodies_data.copy()
        for (bodyID, body_data) in new_bodies_data.items():
            if len(bodies_data) == 1:
                break
            valid_frames = self._get_valid_frames_by_spread(body_data['joints'].reshape(-1, 25, 3))
            num_frames = len(body_data['interval'])
            num_noise = num_frames - len(valid_frames)
            if num_noise == 0:
                continue

            ratio = num_noise / float(num_frames)
            motion = body_data['motion']
            if ratio >= NOISE_SPREAD_THRESHOLD2:  # 0.69754
                del bodies_data[bodyID]
            else:  # Update motion
                joints = body_data['joints'].reshape(-1, 25, 3)[valid_frames]
                body_data['motion'] = min(motion, np.sum(np.var(joints.reshape(-1, 3), axis=0)))

        return bodies_data
    
    def _denoise_bodies_data(self, bodies_data: dict) -> List[Tuple[str, dict]]:
        """
        Denoising data based on some heuristic methods, not necessarily correct for all samples.
        
        Args:
            bodies_data (dict): bodies data

        Returns:
            List[Tuple[str, dict]]: list of (body_id, body_data)
        """
        
        bodies_data = bodies_data['data']
        
        # Step 1: Denoising based on frame length.
        bodies_data = self._denoise_by_length(bodies_data)
        if len(bodies_data) == 1:
            return bodies_data.items()
        
        # Step 2: Denoising based on spread.
        bodies_data = self._denoise_by_spread(bodies_data)
        if len(bodies_data) == 1:
            return bodies_data.items()
        
        bodies_motion = {}
        for (body_id, body_data) in bodies_data.items():
            bodies_motion[body_id] = body_data['motion']
        
        # Sort bodies based on the motion
        bodies_motion = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)
        denoised_bodies_data = []
        for (body_id, _) in bodies_motion:
            denoised_bodies_data.append((body_id, bodies_data[body_id]))
        
        return denoised_bodies_data
        
    def _get_two_actors_points(self, bodies_data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the first and second actor's joints positions and colors locations.
        
        Args:
            bodies_adata (dict): 3 key-value pairs: 'name', 'data', 'num_frames'.
            bodies_data['data'] is also a dict, while the key is bodyID, the value is
            the corresponding body_data which is also a dict with 4 keys:
            - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
            - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
            - interval: a list which records the frame indices.
            - motion: motion amount

        Returns:
            Tuple[np.ndarray, np.ndarray]: joints and colors
        """
        
        num_frames = bodies_data['num_frames']
        
        bodies_data = self._denoise_bodies_data(bodies_data)
        bodies_data = list(bodies_data)
        
        if len(bodies_data) == 1:
            _, body_data = bodies_data[0]
            joints, colors = self._get_one_actor_points(body_data, num_frames)
        else: 
            joints = np.zeros((num_frames, 150), dtype=np.float32)
            colors = np.ones((num_frames, 2, 25, 2), dtype=np.float32) * np.nan

            _, actor1 = bodies_data[0]  # the 1st actor with largest motion
            start1, end1 = actor1['interval'][0], actor1['interval'][-1]
            joints[start1:end1 + 1, :75] = actor1['joints'].reshape(-1, 75)
            colors[start1:end1 + 1, 0] = actor1['colors']
            del bodies_data[0]

            start2, end2 = [0, 0]  # initial interval for actor2 (virtual)

            while len(bodies_data) > 0:
                _, actor = bodies_data[0]
                start, end = actor['interval'][0], actor['interval'][-1]
                if min(end1, end) - max(start1, start) <= 0:  # no overlap with actor1
                    joints[start:end + 1, :75] = actor['joints'].reshape(-1, 75)
                    colors[start:end + 1, 0] = actor['colors']
                    # Update the interval of actor1
                    start1 = min(start, start1)
                    end1 = max(end, end1)
                elif min(end2, end) - max(start2, start) <= 0:  # no overlap with actor2
                    joints[start:end + 1, 75:] = actor['joints'].reshape(-1, 75)
                    colors[start:end + 1, 1] = actor['colors']
                    # Update the interval of actor2
                    start2 = min(start, start2)
                    end2 = max(end, end2)
                del bodies_data[0]
        
        return joints, colors
    
    def _remove_missing_frames(self, 
                               joints: np.ndarray, 
                               colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cut off missing frames which all joints positions are 0s.
        
        Args:
            joints (np.ndarray): joints
            colors (np.ndarray): colors

        Returns:
            Tuple[np.ndarray, np.ndarray]: joints and colors
        """

        # Find valid frame indices that the data is not missing or lost
        # For two-subjects action, this means both data of actor1 and actor2 is missing.
        valid_indices = np.where(joints.sum(axis=1) != 0)[0]  # 0-based index
        missing_indices = np.where(joints.sum(axis=1) == 0)[0]
        num_missing = len(missing_indices)

        if num_missing > 0:  # Update joints and colors
            joints = joints[valid_indices]
            colors[missing_indices] = np.nan

        return joints, colors
        
    def _denoise_sequence(self, bodies_data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get denoised data (joints positions and color locations) from raw skeleton sequences.

        For each frame of a skeleton sequence, an actor's 3D positions of 25 joints represented
        by an 2D array (shape: 25 x 3) is reshaped into a 75-dim vector by concatenating each
        3-dim (x, y, z) coordinates along the row dimension in joint order. Each frame contains
        two actor's joints positions constituting a 150-dim vector. If there is only one actor,
        then the last 75 values are filled with zeros. Otherwise, select the main actor and the
        second actor based on the motion amount. Each 150-dim vector as a row vector is put into
        a 2D numpy array where the number of rows equals the number of valid frames. All such
        2D arrays are put into a list and finally the list is serialized into a cPickle file.
        """
        
        num_bodies = len(bodies_data['data'])
        if num_bodies == 1: 
            num_frames = bodies_data['num_frames']
            body_data = list(bodies_data['data'].values())[0]
            joints, colors = self._get_one_actor_points(body_data, num_frames)
        else:
            joints, colors = self._get_two_actors_points(bodies_data)
            # Remove missing frames
            joints, colors = self._remove_missing_frames(joints, colors)
            num_frames = joints.shape[0]
        
        return joints, colors
    
    def _translate_sequence(self, joints: np.ndarray) -> np.ndarray:
        num_frames = joints.shape[0]
        num_bodies = 1 if joints.shape[1] == 75 else 2
        if num_bodies == 2:
            missing_frames_1 = np.where(joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            if num_bodies == 1:
                joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                joints[f] -= np.tile(origin, 50)

        if (num_bodies == 2) and (cnt1 > 0):
            joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        return joints
    
    def _align_frames(self, joints: np.ndarray) -> np.ndarray:
        
        T, V = joints.shape
        if V == 75:
            new_joints = np.zeros((T, 150))
            new_joints[:] = np.hstack((joints, joints))
            joints = new_joints
        
        # (T, M, V, C)
        joints = joints.reshape((T, 2, 25, 3))
        # (M, C, V, T)
        joints = joints.transpose(1, 3, 2, 0)
        joints = torch.from_numpy(joints).to(dtype=torch.float32)
        joints: torch.Tensor = F.interpolate(joints, (25, self.config.num_frames), mode='bilinear', align_corners=False)
        
        joints: np.ndarray = joints.numpy()
        # (T, C, V, M)
        joints = joints.transpose(1, 3, 2, 0)
        return joints
            
    def _get_sequence_joints(self, folder: str, filename: str) -> np.ndarray:
        """
        Args:
            folder (str): _description_
            filename (str): _description_

        Returns:
            np.ndarray: joints with shape (C, T, V, M)
        """
        bodies_data = self._get_sequence_raw_bodies(folder, filename)
        joints, _ = self._denoise_sequence(bodies_data)
        joints = self._translate_sequence(joints)
        joints = self._align_frames(joints)
        
        return joints
    
    def _get_mean_map(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N, C, T, V, M = data.shape
        mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
        
        return mean_map, std_map
    
    def _save_mean_variance(self, data: np.ndarray, phase: str):
        N, C, T, V, M = data.shape
        joints = np.zeros((N, C * 3, T, V, M))
        bones = np.zeros((N, C * 3, T, V, M))
        
        res = {
            'joints': { 'x': {}, 'y': {}, 'z': {} }, 
            'bones': { 'x': {}, 'y': {}, 'z': {} } 
        }
        
        # joints coordinates
        joints[:, :C] = data
        xj_coord_mean, yj_coord_mean, zj_coord_mean = joints[:, :C].mean(axis=(0, 2, 3, 4)).tolist()
        xj_coord_std, yj_coord_std, zj_coord_std = joints[:, :C].std(axis=(0, 2, 3, 4)).tolist()
        
        res['joints']['x']['coordinate'] = { 'mean': xj_coord_mean, 'std': xj_coord_std }
        res['joints']['y']['coordinate'] = { 'mean': yj_coord_mean, 'std': yj_coord_std }
        res['joints']['z']['coordinate'] = { 'mean': zj_coord_mean, 'std': zj_coord_std }
    
        # joints velocity
        joints[:, C:C*2, :-1] = joints[:, :C, 1:] - joints[:, :C, :-1]
        xj_vel_mean, yj_vel_mean, zj_vel_mean = joints[:, C:C*2].mean(axis=(0, 2, 3, 4)).tolist()
        xj_vel_std, yj_vel_std, zj_vel_std = joints[:, C:C*2].std(axis=(0, 2, 3, 4)).tolist()
        
        res['joints']['x']['velocity'] = { 'mean': xj_vel_mean, 'std': xj_vel_std }
        res['joints']['y']['velocity'] = { 'mean': yj_vel_mean, 'std': yj_vel_std }
        res['joints']['z']['velocity'] = { 'mean': zj_vel_mean, 'std': zj_vel_std }
        
        # joints distance to center
        joints[:, C*2:] = joints[:, :C] - np.expand_dims(joints[:, :C, :, 1], 3)
        xj_dis_mean, yj_dis_mean, zj_dis_mean = joints[:, C*2:].mean(axis=(0, 2, 3, 4)).tolist()
        xj_dis_std, yj_dis_std, zj_dis_std = joints[:, C*2:].std(axis=(0, 2, 3, 4)).tolist()
        
        res['joints']['x']['distance'] = { 'mean': xj_dis_mean, 'std': xj_dis_std }
        res['joints']['y']['distance'] = { 'mean': yj_dis_mean, 'std': yj_dis_std }
        res['joints']['z']['distance'] = { 'mean': zj_dis_mean, 'std': zj_dis_std }
        
        # bones
        skeleton = self.config.to_skeleton_graph()
        conn = skeleton.joints_connections
        for u, v in conn:
            bones[:, :C, :, u] = joints[:, :C, :, u] - joints[:, :C, :, v]
        
        # bones coordinates
        xb_coord_mean, yb_coord_mean, zb_coord_mean = bones[:, :C].mean(axis=(0, 2, 3, 4)).tolist()
        xb_coord_std, yb_coord_std, zb_coord_std = bones[:, :C].std(axis=(0, 2, 3, 4)).tolist()
        
        res['bones']['x']['coordinate'] = { 'mean': xb_coord_mean, 'std': xb_coord_std }
        res['bones']['y']['coordinate'] = { 'mean': yb_coord_mean, 'std': yb_coord_std }
        res['bones']['z']['coordinate'] = { 'mean': zb_coord_mean, 'std': zb_coord_std }
        
        # bones velocity
        bones[:, C:C*2, :-1] = bones[:, :C, 1:] - bones[:, :C, :-1]
        xb_vel_mean, yb_vel_mean, zb_vel_mean = bones[:, C:C*2].mean(axis=(0, 2, 3, 4)).tolist()
        xb_vel_std, yb_vel_std, zb_vel_std = bones[:, C:C*2].std(axis=(0, 2, 3, 4)).tolist()
        
        res['bones']['x']['velocity'] = { 'mean': xb_vel_mean, 'std': xb_vel_std }
        res['bones']['y']['velocity'] = { 'mean': yb_vel_mean, 'std': yb_vel_std }
        res['bones']['z']['velocity'] = { 'mean': zb_vel_mean, 'std': zb_vel_std }
        
        # bones angle
        bone_length = 0
        for c in range(C):
            bone_length += bones[:, c] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for c in range(C):
            bones[:, C*2+c] = np.arccos(bones[:, c] / bone_length)
        
        xb_ang_mean, yb_ang_mean, zb_ang_mean = bones[:, C*2:].mean(axis=(0, 2, 3, 4)).tolist()
        xb_ang_std, yb_ang_std, zb_ang_std = bones[:, C*2:].std(axis=(0, 2, 3, 4)).tolist()
        
        res['bones']['x']['angle'] = { 'mean': xb_ang_mean, 'std': xb_ang_std }
        res['bones']['y']['angle'] = { 'mean': yb_ang_mean, 'std': yb_ang_std }
        res['bones']['z']['angle'] = { 'mean': zb_ang_mean, 'std': zb_ang_std }
        
        # save results
        file = os.path.join(self.config.dataset_path, f'{phase}_mean_std.yaml')
        self.logger.info(f'Saving mean and std of data in {file}')
        with open(file, 'w') as f:
            yaml.dump(res, f)
                            
    def _gendata(self, files: List[Tuple[str, str]], phase: str): 
        data = []
        labels = []

        for (folder, filename) in tqdm(files, desc=f'{self.config.name}-{phase}'):
            action_loc = filename.find('A')
            action_class = int(filename[(action_loc+1):(action_loc+4)])
            
            joints = self._get_sequence_joints(folder, filename)
            data.append(joints)
            labels.append(action_class - 1) # to 0-indexed
        
        # Save joints
        data = np.array(data)
        data_file = os.path.join(self.config.dataset_path, f'{phase}_data.npy')
        self.logger.info(f'Saving skeletons data in {data_file}')
        np.save(data_file, data)
        
        # Save labels
        labels = np.array(labels)
        labels_file = os.path.join(self.config.dataset_path, f'{phase}_labels.npy')
        self.logger.info(f'Saving labels data in {labels_file}')
        np.save(labels_file, labels)
        
        # Save mean and std
        self._save_mean_variance(data, phase)
    
    def _get_train_test_split(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        train = []
        test = []
        
        training_samples = self._get_training_samples()
        
        for (folder, filename) in self.file_list:
            
            setup_loc = filename.find('S')
            camera_loc = filename.find('C')
            subject_loc = filename.find('P')
            setup_id = int(filename[(setup_loc+1):(setup_loc+4)])
            camera_id = int(filename[(camera_loc+1):(camera_loc+4)])
            subject_id = int(filename[(subject_loc+1):(subject_loc+4)])
            
            if self.config.name == 'ntu60-xview':
                is_training_sample = (camera_id in training_samples)
            elif self.config.name == 'ntu60-xsub' or self.config.name == 'ntu120-xsub':
                is_training_sample = (subject_id in training_samples)
            elif self.config.name == 'ntu120-xset':
                is_training_sample = (setup_id in training_samples)
            else: # this should never happen
                e = ValueError(f'Dataset {self.config.name} does not exist.')
                if self.config.debug:
                    self.logger.exception(e)
                else: 
                    self.logger.error(e)
                raise e
            
            if is_training_sample:
                train.append((folder, filename))
            else:
                test.append((folder, filename))
        
        return train, test
            
    def start(self):
        train_files, test_files = self._get_train_test_split()
        
        if self.config.debug:
            train_files = train_files[:300]
            test_files = test_files[:300]
        
        self.logger.info(f'Generating training data of {self.config.name} dataset')
        self._gendata(train_files, 'train')
        
        self.logger.info(f'Generating test data of {self.config.name} dataset')
        self._gendata(test_files, 'test')
        