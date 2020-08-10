
import pandas
from torch.utils.data import Dataset
import os
import numpy as np


class Semantic3dDataset(Dataset):
    def __init__(self, data_root, num_points, caching=False):
        self.data_root = data_root
        self.num_points = num_points
        self.caching = caching
        self.data = []
        self.load_data()

    def load_data(self):
        for root, dirs, files in os.walk(self.data_root):
            for point_file in files:
                if point_file.endswith('.txt'):
                    item = {'point_file': os.path.join(root, point_file),
                            'label_file': None, 'points': None, 'labels': None}
                    file_name = os.path.splitext(point_file)[0]
                    item['file_name'] = file_name
                    label_file = file_name + '.labels'
                    if label_file in files:
                        item['label_file'] = os.path.join(root, label_file)
                    self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        points = self.data[idx]['points']
        if points is None:
            print('read', self.data[idx]['point_file'])
            points = pandas.read_csv(self.data[idx]['point_file'], delimiter=" ", header=None, dtype='float32')
            points = points.to_numpy()
            if self.caching:
                self.data[idx]['points'] = points

        labels = self.data[idx]['labels']
        if labels is None and self.data[idx]['label_file'] is not None:
            print('read', self.data[idx]['label_file'])
            labels = pandas.read_csv(self.data[idx]['label_file'], header=None, dtype='int64')
            labels = labels.to_numpy()
            if self.caching:
                self.data[idx]['labels'] = labels

        if labels is not None:
            assert points.shape[0] == labels.shape[0]

        while points.shape[0] < self.num_points:
            points = np.concatenate([points, points])
            if labels is not None:
                labels = np.concatenate([labels, labels])

        if self.num_points > 0:
            choices = np.random.choice(points.shape[0], self.num_points)
            points = points[choices]
            if labels is not None:
                labels = labels[choices]
        sample = {'points': points, 'name':self.data[idx]['file_name']}
        if labels is not None:
            sample['labels'] = labels

        # choice = np.random.choice(points.shape[0], 1000000)
        # np.savetxt('test.txt', self.data[idx]['points'][choice], fmt='%.5f')
        # np.savetxt('test.labels', self.data[idx]['labels'][choice], fmt='%.d')
        # np.savetxt(self.data[idx]['file_name'] + '-%d' % self.num_points + '.txt', points, fmt='%.5f')
        # np.savetxt(self.data[idx]['file_name'] + '-%d' % self.num_points + '.labels', labels, fmt='%.d')

        return sample

