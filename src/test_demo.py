import argparse
import os
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from model import UGC_BVQA_model
from pytorchvideo.models.hub import slowfast_r50
import cv2
from PIL import Image
from torchvision import transforms

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)


def video_processing_spatial(dist):
    video_name = dist
    if not os.path.exists(video_name):
        logger.error(f"Video file {video_name} does not exist.")
        return None, None

    video_capture = cv2.VideoCapture(video_name)
    cap = cv2.VideoCapture(video_name)

    video_channel = 3
    video_height_crop = 448
    video_width_crop = 448
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    video_length_read = int(video_length / video_frame_rate)

    transformations = transforms.Compose([
        transforms.Resize(520),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
    video_read_index = 0
    frame_idx = 0

    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if not has_frames:
            break

        if (video_read_index < video_length_read) and (frame_idx % video_frame_rate == 0):
            read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            read_frame = transformations(read_frame)
            transformed_video[video_read_index] = read_frame
            video_read_index += 1

        frame_idx += 1

    if video_read_index < video_length_read:
        for i in range(video_read_index, video_length_read):
            transformed_video[i] = transformed_video[video_read_index - 1]

    video_capture.release()

    return transformed_video, video_name


def pack_pathway_output(frames, device):
    fast_pathway = frames
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(0, frames.shape[2] - 1, frames.shape[2] // 4).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]
    return frame_list


class SlowFast(torch.nn.Module):
    def __init__(self):
        super(SlowFast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)
            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])
            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)
        return slow_feature, fast_feature


def video_processing_motion(dist):
    video_name = dist
    if not os.path.exists(video_name):
        logger.error(f"Video file {video_name} does not exist.")
        return None, None

    video_capture = cv2.VideoCapture(video_name)
    cap = cv2.VideoCapture(video_name)

    video_channel = 3
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    video_clip = int(video_length / video_frame_rate)
    video_clip_min = 8
    video_length_clip = 32

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    transformed_frame_all = []
    video_read_index = 0

    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if not has_frames:
            break

        read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        read_frame = transform(read_frame)
        transformed_frame_all.append(read_frame)
        video_read_index += 1

    video_capture.release()

    transformed_video_all = []
    for i in range(video_clip):
        transformed_video = torch.stack(
            transformed_frame_all[i * video_frame_rate:(i * video_frame_rate + video_length_clip)])
        if len(transformed_video) < video_length_clip:
            padding = torch.stack([transformed_video[-1]] * (video_length_clip - len(transformed_video)))
            transformed_video = torch.cat([transformed_video, padding], dim=0)
        transformed_video_all.append(transformed_video)

    if video_clip < video_clip_min:
        for i in range(video_clip, video_clip_min):
            transformed_video_all.append(transformed_video_all[video_clip - 1])

    return transformed_video_all, video_name


def main(config):
    try:
        logger.info("Configurations: %s", config)
        device = torch.device('cuda' if config.is_gpu else 'cpu')
        logger.info('Using device: %s', device)

        # 设置随机种子
        set_seed(42)

        model_motion = SlowFast().to(device)
        model = UGC_BVQA_model.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model).to(device)

        # 加载模型权重文件 - 使用实际存在的模型文件
        model_path = '../ckpts/UGC_BVQA_model_LSVQ_L1RankLoss_NR_v0_epoch_12_SRCC_1.000000.pth'
        if not os.path.exists(model_path):
            # 如果默认模型不存在，尝试使用其他可用的模型
            available_models = [
                '../ckpts/UGC_BVQA_model.pth',
                '../ckpts/UGC_BVQA_model_LSVQ_L1RankLoss_NR_v0_epoch_20_SRCC_0.960296.pth'
            ]
            for alt_model in available_models:
                if os.path.exists(alt_model):
                    model_path = alt_model
                    logger.info(f"Using alternative model: {model_path}")
                    break
            else:
                logger.error("No model file found in ckpts directory")
                return
        
        logger.info(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        # 假设权重文件中的键不包含 'module.' 前缀，而模型定义中包含
        new_state_dict = {f'module.{k}': v for k, v in state_dict.items()}

        # 使用 strict=True 加载模型权重
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()

        if config.method_name == 'single-scale':
            video_dist_spatial, video_name = video_processing_spatial(config.dist)
            if video_dist_spatial is None or video_name is None:
                logger.error("Failed to process spatial features.")
                return

            video_dist_motion, _ = video_processing_motion(config.dist)
            if video_dist_motion is None:
                logger.error("Failed to process motion features.")
                return

            with torch.no_grad():
                video_dist_spatial = video_dist_spatial.unsqueeze(dim=0).to(device)
                n_clip = len(video_dist_motion)
                feature_motion = torch.zeros([n_clip, 2048 + 256]).to(device)

                for idx, ele in enumerate(video_dist_motion):
                    ele = ele.unsqueeze(dim=0).permute(0, 2, 1, 3, 4)
                    ele = pack_pathway_output(ele, device)
                    ele_slow_feature, ele_fast_feature = model_motion(ele)

                    ele_feature_motion = torch.cat([ele_slow_feature.squeeze(), ele_fast_feature.squeeze()])
                    feature_motion[idx] = ele_feature_motion

                feature_motion = feature_motion.unsqueeze(dim=0)
                outputs = model(video_dist_spatial, feature_motion)
                y_val = outputs.item()

                logger.info('The video name: %s', video_name)
                logger.info('The quality score: %.4f', y_val)

            output_name = config.output
            with open(output_name, 'w') as f:
                f.write(f'{video_name},{y_val}\n')

    except Exception as e:
        logger.error("An error occurred: %s", str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_name', type=str, default='single-scale')
    parser.add_argument('--dist', type=str, required=True, help='Path to the video file')
    parser.add_argument('--output', type=str, default='output.txt', help='Output file name')
    parser.add_argument('--is_gpu', action='store_true', help='Use GPU if available')

    config = parser.parse_args()
    main(config)