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
parser.add_argument('--test_dir', type=str, default='data/Semantic3D', help='root dir of dataset')
parser.add_argument('--out_dir', type=str, default='out', help='out dir')
parser.add_argument('--model_dir', type=str, default='save_model', help='path of save model')
parser.add_argument('--model_name', type=str, default='001', help='path of write summary')
parser.add_argument('--num_classes', type=int, default=9, help='number of classes')
parser.add_argument('--num_points', type=int, default=10000000, help='number of points')
parser.add_argument('--epochs', type=int, default=10000, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=2, help='number of batch size')
parser.add_argument('--summary_frequency', type=int, default=1, help='print_steps')
parser.add_argument('--save_frequency', type=int, default=10)
parser.add_argument('--test_frequency', type=int, default=10)
parser.add_argument('--num_planes', type=int, default=3, help='number of image planes')
parser.add_argument('--resume', type=int, default=1, help='resume train')
parser.add_argument('--caching', type=int, default=0, help='resume train')
parser.add_argument('--gpu_id', type=str, default='0', help='gpu_id')
parser.add_argument('--width', type=int, default=512, help='gpu_id')
parser.add_argument('--height', type=int, default=512, help='gpu_id')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def summary_thick_seg(thick_seg, writer, flag, step):
    image = thick_seg.planes[..., 0, 1:4]
    rho = thick_seg.planes[..., 0, 6:7]
    semantic = (thick_seg.planes[..., 0, 7] * args.num_classes).long()
    predict = torch.argmax(thick_seg.predict2d, dim=-1)[..., 0]

    writer.add_images(flag + '/image', image, global_step=step, dataformats='NHWC')
    writer.add_images(flag + '/rho', rho, global_step=step, dataformats='NHWC')
    writer.add_images(flag + '/semantic', util.visualize(semantic), global_step=step, dataformats='NHWC')
    writer.add_images(flag + '/predict', util.visualize(predict), global_step=step, dataformats='NHWC')

    if thick_seg.labels2d is not None:
        labels = thick_seg.labels2d[..., 0]
        writer.add_images(flag + '/labels', util.visualize(labels), global_step=step, dataformats='NHWC')
        writer.add_scalar(flag + '/os', thick_seg.oa, global_step=step)
        writer.add_scalar(flag + '/mean_iou', thick_seg.mean_iou, global_step=step)
        writer.add_histogram(flag + '/iou', thick_seg.iou, global_step=step)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = Semantic3dDataset(args.data_dir, num_points=args.num_points, caching=args.caching > 0)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(train_data, batch_size=1, shuffle=False, drop_last=False)

    test_data = Semantic3dDataset(args.test_dir, num_points=0, caching=args.caching > 0)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)

    thick_seg = ThickSeg3D(num_planes=args.num_planes, classes=args.num_classes)
    thick_seg.train()
    thick_seg.to(device)

    save_dir = os.path.join(args.model_dir, args.model_name)
    model_path = os.path.join(save_dir, 'model.pth')
    if args.resume > 0 and os.path.exists(model_path):
        thick_seg.net.load_state_dict(torch.load(model_path))

    date_time = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    writer = SummaryWriter(os.path.join('runs', args.model_name + date_time))
    writer.add_text('args', str(args), 0)

    val_step = 0
    test_step = 0
    for epoch in range(args.epochs):
        oaes = []
        mIoUs = []
        ious = []
        if epoch % args.test_frequency == 0:
            thick_seg.eval()
            thick_seg.plane_width = args.width * 2
            thick_seg.plane_height = args.height
            with torch.no_grad():
                for data in val_loader:
                    thick_seg(data)
                    summary_thick_seg(thick_seg, writer, 'val', val_step)
                    val_step += 1
                for data in test_loader:
                    thick_seg(data)
                    summary_thick_seg(thick_seg, writer, 'test', test_step)
                    test_step += 1

        thick_seg.train()
        thick_seg.plane_width = args.width
        thick_seg.plane_height = args.height
        for data in train_loader:
            thick_seg(data)
            oaes.append(thick_seg.oa)
            mIoUs.append(thick_seg.mean_iou)
            ious.append(thick_seg.iou.cpu().numpy())

        if epoch % args.summary_frequency == 0:
            loss = sum(thick_seg.loss_array) / len(thick_seg.loss_array)
            iou = np.stack(ious, axis=0)
            writer.add_histogram('train/loss', np.array(thick_seg.loss_array), global_step=epoch)
            writer.add_histogram('train/acc', np.array(thick_seg.acc_array), global_step=epoch)
            writer.add_histogram('train/hit_rate', np.array(thick_seg.hit_rate_array), global_step=epoch)
            writer.add_histogram('train/iou', iou.mean(axis=0), global_step=epoch)
            writer.add_scalar('train/oa', sum(oaes) / len(oaes), global_step=epoch)
            writer.add_scalar('train/mIoU', sum(mIoUs) / len(mIoUs), global_step=epoch)

            image = thick_seg.planes[..., 0, 1:4]
            rho = thick_seg.planes[..., 0, 6:7]
            semantic = (thick_seg.planes[..., 0, 7] * args.num_classes).long()
            predict = torch.argmax(thick_seg.predict2d, dim=-1)[..., 0]
            labels = thick_seg.labels2d[..., 0]

            writer.add_images('train/image', image, global_step=epoch, dataformats='NHWC')
            writer.add_images('train/rho', rho, global_step=epoch, dataformats='NHWC')
            writer.add_images('train/semantic', util.visualize(semantic), global_step=epoch, dataformats='NHWC')
            writer.add_images('train/predict', util.visualize(predict), global_step=epoch, dataformats='NHWC')
            writer.add_images('train/labels', util.visualize(labels), global_step=epoch, dataformats='NHWC')

        print(epoch, 'loss', loss)

        if (epoch + 1) % args.save_frequency == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(thick_seg.net.state_dict(), model_path)
            # print('saving points')
            # thick_seg.save_points(data, args.out_dir)


if __name__ == '__main__':
    main()
