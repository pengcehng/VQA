# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import torch
from data_loader import VideoDataset_NR_LSVQ_SlowFast_feature
from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn
import gc  # 导入垃圾回收模块

def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """

    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list


class SlowFastModel(torch.nn.Module):
    def __init__(self):
        super(SlowFastModel, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0, 5):
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


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SlowFastModel()
    model = model.to(device)

    resize = config.resize

    ## training data
    videos_dir = 'LSVQ'
    datainfo_train = 'data/LSVQ_whole_train.csv'
    datainfo_test = 'data/LSVQ_whole_test.csv'
    datainfo_test_1080p = 'data/LSVQ_whole_test_1080p.csv'
    transformations_test = transforms.Compose([
        transforms.Resize([resize, resize]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    column_names = ['name', 'mos']

    trainset = VideoDataset_NR_LSVQ_SlowFast_feature(videos_dir, datainfo_train, transformations_test, resize,
                                                     column_names=column_names)
    testset = VideoDataset_NR_LSVQ_SlowFast_feature(videos_dir, datainfo_test, transformations_test, resize,
                                                    column_names=column_names)
    testset_1080p = VideoDataset_NR_LSVQ_SlowFast_feature(videos_dir, datainfo_test_1080p, transformations_test, resize,
                                                          column_names=column_names, is_test_1080p=True)

    ## dataloader
    # 减少 num_workers 以减少内存占用
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    test_loader_1080p = torch.utils.data.DataLoader(testset_1080p, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # do validation after each epoch
    with torch.no_grad():
        model.eval()

        for loader, dataset in zip([train_loader, test_loader, test_loader_1080p], [trainset, testset, testset_1080p]):
            for i, (video, mos) in enumerate(loader):
                video_name = dataset.get_video_name(i)  # 获取视频名称
                feature_dir = os.path.join(config.feature_save_folder, video_name)
                if not os.path.exists(feature_dir):
                    os.makedirs(feature_dir)

                for idx, ele in enumerate(video):
                    ele = ele.permute(0, 2, 1, 3, 4)
                    inputs = pack_pathway_output(ele, device)
                    slow_feature, fast_feature = model(inputs)
                    np.save(os.path.join(feature_dir, f'feature_{idx}_slow_feature.npy'), slow_feature.cpu().numpy())
                    np.save(os.path.join(feature_dir, f'feature_{idx}_fast_feature.npy'), fast_feature.cpu().numpy())

                print(f"Processed {video_name}")  # 只在每个视频处理完后打印一次
                del video, mos, ele, inputs, slow_feature, fast_feature  # 删除变量
                gc.collect()  # 强制垃圾回收


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_workers', type=int, default=2)  # 减少工作线程数量
    parser.add_argument('--resize', type=int, default=64)  # 减小分辨率以节省内存
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--feature_save_folder', type=str, default='ckpts')

    config = parser.parse_args()

    main(config)
