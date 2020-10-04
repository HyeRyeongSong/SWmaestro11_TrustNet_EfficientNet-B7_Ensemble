import argparse
import os
import glob
import yaml
import pickle
import tqdm

import torch
from torch.utils.data import DataLoader

#from dsfacedetector.face_ssd_infer import SSD
from datasets import UnlabeledVideoDataset
from facenet_pytorch import MTCNN
from os import cpu_count
import cv2
from PIL import Image
import numpy as np

DETECTOR_WEIGHTS_PATH = 'external_data/20180402-114759-vggface2-features.pth'
DETECTOR_THRESHOLD = 0.3
DETECTOR_STEP = 6
DETECTOR_TARGET_SIZE = (512, 512)

BATCH_SIZE = 1 
NUM_WORKERS = 0

DETECTIONS_ROOT = 'detections'
DETECTIONS_FILE_NAME = 'detections.pkl'

##my code

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

##end

def main():
    parser = argparse.ArgumentParser(description='Detects faces on videos')
    parser.add_argument('--num_parts', type=int, default=1, help='Number of parts')
    parser.add_argument('--part', type=int, default=0, help='Part index')

    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    content = []
    for path in glob.iglob(os.path.join(config['DFDC_DATA_PATH'], 'dfdc_train_part_*', '*.mp4')):
        parts = path.split('/')
        content.append('/'.join(parts[-2:]))
    content = sorted(content)

    print('Total number of videos: {}'.format(len(content)))

    part_size = len(content) // args.num_parts + 1
    assert part_size * args.num_parts >= len(content)
    part_start = part_size * args.part
    part_end = min(part_start + part_size, len(content))
    print('Part {} ({}, {})'.format(args.part, part_start, part_end))

    dataset = UnlabeledVideoDataset(config['DFDC_DATA_PATH'], content[part_start:part_end])

    detector = MTCNN(post_process=False, select_largest=False, device='cuda')
    #state = torch.load(DETECTOR_WEIGHTS_PATH, map_location=lambda storage, loc: storage)
    #detector.load_state_dict(state)
    device = torch.device('cuda')
    #detector = detector.eval().to(device)

    #loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=lambda X: X,drop_last=False)
    print("1")
    loader = DataLoader(dataset, shuffle=False, num_workers=cpu_count() - 2, batch_size=BATCH_SIZE, collate_fn=lambda X: X, drop_last=False)

    dst_root = os.path.join(config['ARTIFACTS_PATH'], DETECTIONS_ROOT)
    os.makedirs(dst_root, exist_ok=True)

    print("2")
    for video_sample in tqdm.tqdm(loader):
        frames = video_sample[0]['frames']
        video_idx = video_sample[0]['index']
        video_rel_path = dataset.content[video_idx]

        detections = []
        print("3")
        for frame in frames[::DETECTOR_STEP]:
            print("4")
            print(type(frame))
            result = detector.detect(frame, landmarks=False)
            print(result)
            #boxes, probs = detector.detect(frame, landmarks=False)
            print("5")
            #detections.append({'boxes': boxes, 'scores': probs})
            '''
            with torch.no_grad():
                print("4")
                boxes, probs = detector.detect(frame, landmarks=False)
                print("5")
                detections.append({'boxes': boxes, 'scores': probs})
            '''

        os.makedirs(os.path.join(dst_root, video_rel_path), exist_ok=True)
        with open(os.path.join(dst_root, video_rel_path, DETECTIONS_FILE_NAME), 'wb') as f:
            pickle.dump(detections, f)


if __name__ == '__main__':
    main()
