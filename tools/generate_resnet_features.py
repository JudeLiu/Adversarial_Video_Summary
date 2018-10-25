import numpy as np
import torch
from torchvision.models import resnet101
from torchvision import transforms
import os
import imageio
import h5py 
from skimage.transform import resize
from feature_extraction import ResNetFeature

frames_root_dir = '/slwork/jun/vsum_project/Datasets/VSUMM'
data_type = ['database', 'new_database']
feature_output_dir = '/slwork/jun/vsum_project/Adversarial_Video_Summary/resnet101_features'
resnet_input_shape = (224, 224)
batch_size = 1

class Rescale(object):
    """Rescale a image to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = resize(image, (new_w, new_h))
        return img


class ToTensor(object):
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)/255).permute(2, 0, 1)

            
def main():
    net = ResNetFeature(feature='resnet101').cuda(0)
    resnet_transform = transforms.Compose([
        Rescale(*resnet_input_shape),
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for ty in data_type[0:1]:
        # method 1: use imageio.mimread, cost too many memories
        # features = {}
        # all_video_dir = os.path.join(frames_root_dir, ty)
        # for video in sorted(os.listdir(all_video_dir), key=lambda s:int(s[1:-4])):
        #     print('processing {}'.format(os.path.join(all_video_dir, video)))
        #     frames = imageio.mimread(os.path.join(all_video_dir, video), memtest=False)
        #     frames = np.array(frames)
        #     resize_frames = np.zeros((frames.shape[0], 224, 224, 3))
        #     for i, f in enumerate(frames):
        #         resize_frames[i] = resize(f, resnet_input_shape)
        #     feat = net(torch.from_numpy(resize_frames.astype(np.float32)).permute(0,3,1,2))
        #     features['video'] = feat
        # with h5py.File('database.hdf5','w') as f:
        #     dset = f.create_dataset('database', data=np.array(features.items()))     

        # del frames
        # del resize_frames


        # method 2: use imageio.get_reader
        all_video_dir = os.path.join(frames_root_dir, ty)
        for video in sorted(os.listdir(all_video_dir), key=lambda s:int(s[1:-4])):
            video_name = os.path.join(all_video_dir, video)
            h5file_name = '{}.hdf5'.format(os.path.join(feature_output_dir, video[:-4]))

            print('processing {}'.format(video_name))
            if os.path.exists(h5file_name):
                print('existed, skip')
                continue
            
            reader = imageio.get_reader(video_name)
            print('reader created')
            
            frame_len = reader.get_length()
            print('{} has {} frames'.format(video, frame_len))

            feature = []

            for indices in np.split(np.arange(frame_len), frame_len//batch_size):
                frames = [resnet_transform(reader.get_next_data()) for i in indices]
                frames = torch.stack(frames)
                feat = net(frames.cuda(0)).cpu()
                feature.append(feat)
            feature = torch.cat(feature)

            with h5py.File(h5file_name) as f:
                d = f.create_dataset('feat', data=feature.numpy())
                d.attrs['vname'] = video[:-4]
            


        # method 3: need opencv
        # for ty in data_type[0:1]:
        #     all_frames_dir = os.path.join(frames_root_dir, ty)
            
        #     all_frames = {}
        #     for work_dir in os.listdir(all_frames_dir)[0:1]:
        #         frames = []
        #         wd = os.path.join(frames_root_dir, ty, work_dir)
        #         frames_name = sorted(os.listdir(wd), key=lambda s:int(s[:-4]))
        #         for name in frames_name:
        #             img = imread(os.path.join(wd, name))
        #             frames.append(img)
        #         frames = np.array(frames)
        #         print(frames.shape)

if __name__ == '__main__':
    main()
