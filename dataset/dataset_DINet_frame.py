import torch
import numpy as np
import json
import random
import cv2

from torch.utils.data import Dataset
import os 
import sys 
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取当前脚本所在路径的上层目录
parent_dir = os.path.dirname(current_dir)
print(f'current_dir: {current_dir} parent_dir: {parent_dir}')
sys.path.append(parent_dir)
from config.config import DINetTrainingOptions

def get_data(json_name,augment_num):
    print('start loading data')
    with open(json_name,'r') as f:
        data_dic = json.load(f)
    data_dic_name_list = []
    for augment_index in range(augment_num):
        for video_name in data_dic.keys():
            data_dic_name_list.append(video_name)
    random.shuffle(data_dic_name_list)
    print('finish loading')
    return data_dic_name_list,data_dic


class DINetDataset(Dataset):
    def __init__(self,path_json,augment_num,mouth_region_size):
        super(DINetDataset, self).__init__()
        self.data_dic_name_list,self.data_dic = get_data(path_json,augment_num)
        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size//2
        self.radius_1_4 = self.radius//4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)

    def __getitem__(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])
        random_anchor = random.sample(range(video_clip_num), 6)
        source_anchor, reference_anchor_list = random_anchor[0],random_anchor[1:]
        ## load source image
        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_random_index = random.sample(range(2, 7), 1)[0]
        source_image_data = cv2.imread(source_image_path_list[source_random_index])[:, :, ::-1]
        source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h))/ 255.0
        source_image_mask = source_image_data.copy()
        source_image_mask[self.radius:self.radius+self.mouth_region_size,self.radius_1_4:self.radius_1_4 +self.mouth_region_size ,:] = 0

        ## load deep speech feature
        deepspeech_feature = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'][source_random_index - 2:source_random_index + 3])

        ## load reference images
        reference_frame_data_list = []
        for reference_anchor in reference_anchor_list:
            reference_frame_path_list = self.data_dic[video_name]['clip_data_list'][reference_anchor]['frame_path_list']
            reference_random_index = random.sample(range(9), 1)[0]
            reference_frame_path = reference_frame_path_list[reference_random_index]
            reference_frame_data = cv2.imread(reference_frame_path)[:, :, ::-1]
            reference_frame_data = cv2.resize(reference_frame_data, (self.img_w, self.img_h))/ 255.0
            reference_frame_data_list.append(reference_frame_data)
        reference_clip_data = np.concatenate(reference_frame_data_list, 2)

        # display the source image and reference images
        # display_img = np.concatenate([source_image_data,source_image_mask]+reference_frame_data_list,1)
        # cv2.imshow('image display',(display_img[:,:,::-1] * 255).astype(np.uint8))
        # cv2.waitKey(-1)

        # # to tensor
        source_image_data = torch.from_numpy(source_image_data).float().permute(2,0,1)
        source_image_mask = torch.from_numpy(source_image_mask).float().permute(2,0,1)
        reference_clip_data = torch.from_numpy(reference_clip_data).float().permute(2,0,1)
        deepspeech_feature = torch.from_numpy(deepspeech_feature).float().permute(1,0)
        return source_image_data,source_image_mask, reference_clip_data,deepspeech_feature

    def __len__(self):
        return self.length

if __name__ == '__main__':
   
    opt = DINetTrainingOptions().parse_args()
   
    # load training data in memory
    train_data = DINetDataset(opt.train_data,opt.augment_num,opt.mouth_region_size)
    source_image_data,source_image_mask, reference_clip_data,deepspeech_feature = train_data.__getitem__(0)
   
    print(f'source_image_data shape: {source_image_data.shape}')
    print(f'source_image_mask shape: {source_image_mask.shape}')
    print(f'reference_clip_data shape: {reference_clip_data.shape}')
    print(f'deepspeech_feature shape: {deepspeech_feature.shape}')
    cv2.imwrite('./asserts/tmp/source_img.jpg', source_image_data.permute(1,2,0).numpy() * 255)
    cv2.imwrite('./asserts/tmp/source_img_mask.jpg', source_image_mask.permute(1, 2, 0).numpy() * 255)
    cv2.imwrite('./asserts/tmp/ref_clip.jpg', reference_clip_data.permute(1,2,0).numpy() * 255)


