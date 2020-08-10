# from laspy.file import File
import os
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from data import Semantic3dDataset
import util
from thick_seg import ThickSeg3D
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_dir', type=str, default='data/Semantic3D', help='root dir of dataset')     # Semantic3D
parser.add_argument('--out_dir', type=str, default='out', help='out dir')
parser.add_argument('--model_dir', type=str, default='save_model', help='path of save model')
parser.add_argument('--model_name', type=str, default='test-001', help='path of write summary')
parser.add_argument('--num_classes', type=int, default=9, help='number of classes')
parser.add_argument('--num_points', type=int, default=0, help='number of points')
parser.add_argument('--batch_size', type=int, default=4, help='number of batch size')
parser.add_argument('--num_planes', type=int, default=3, help='number of image planes')
parser.add_argument('--caching', type=int, default=0, help='resume train')
parser.add_argument('--gpu_id', type=str, default='0', help='gpu_id')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = Semantic3dDataset(args.data_dir, num_points=args.num_points, caching=args.caching > 0)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

    thick_seg = ThickSeg3D(num_planes=args.num_planes, classes=args.num_classes, width=1024, height=512)
    thick_seg.eval()
    thick_seg.to(device)

    save_dir = os.path.join(args.model_dir, args.model_name)
    model_path = os.path.join(save_dir, 'model.pth')
    assert os.path.exists(model_path)
    thick_seg.net.load_state_dict(torch.load(model_path))

    date_time = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    writer = SummaryWriter(os.path.join('runs', args.model_name + date_time))
    writer.add_text('args', str(args), 0)

    step = 0
    for data in train_loader:
        start = datetime.now()
        thick_seg(data)
        elapsed = datetime.now() - start
        print('use', elapsed.seconds, 'seconds')
        print('saving points')
        thick_seg.save_points(data, os.path.join(args.out_dir, args.model_name))
        print('saving labels')
        thick_seg.save_labels(data, os.path.join(args.out_dir, args.model_name))

        image = thick_seg.planes[..., 0, 1:4]
        rho = thick_seg.planes[..., 0, 6:7]
        semantic = (thick_seg.planes[..., 0, 7] * args.num_classes).long()
        predict = torch.argmax(thick_seg.predict2d, dim=-1)[..., 0]

        writer.add_images('train/image', image, global_step=step, dataformats='NHWC')
        writer.add_images('train/rho', rho, global_step=step, dataformats='NHWC')
        writer.add_images('train/semantic', util.visualize(semantic), global_step=step, dataformats='NHWC')
        writer.add_images('train/predict', util.visualize(predict), global_step=step, dataformats='NHWC')

        if thick_seg.labels2d is not None:
            labels = thick_seg.labels2d[..., 0]
            writer.add_images('train/labels', util.visualize(labels), global_step=step, dataformats='NHWC')

        step += 1


if __name__ == '__main__':
    main()
