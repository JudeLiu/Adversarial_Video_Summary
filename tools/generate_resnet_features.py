import numpy as np
import torch
from torchvision.models import resnet101
from torchvision import transforms
import os
import imageio
import h5py 
from skimage.transform import resize
from feature_extraction import ResNetFeature
from imageio import imread
from time import time

frames_root_dir = '/slwork/jun/vsum_project/Datasets/VSUMM'
data_type = ['database', 'new_database']
feature_output_dir = '/slwork/jun/vsum_project/Adversarial_Video_Summary/resnet101_features'
resnet_input_shape = (224, 224)
batch_size = 128
use_gpu = 1
devices = 'cuda:0' if use_gpu else 'cpu'


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
    net = ResNetFeature(feature='resnet101').to(device=devices)
    resnet_transform = transforms.Compose([
        Rescale(*resnet_input_shape),
        ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for ty in data_type:
        # method 1: use imageio.mimread, cost too many memories, failed
        #
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


        # method 2: use imageio.get_reader, something wrong with the video decoder, cann't fix it
        #
        # all_video_dir = os.path.join(frames_root_dir, ty)
        # for video in sorted(os.listdir(all_video_dir), key=lambda s:int(s[1:-4])):
        #     video_name = os.path.join(all_video_dir, video)
        #     h5file_name = '{}.hdf5'.format(os.path.join(feature_output_dir, video[:-4]))

        #     print('processing {}'.format(video_name))
        #     if os.path.exists(h5file_name):
        #         print('existed, skip')
        #         continue
            
        #     reader = imageio.get_reader(video_name)
        #     print('reader created')
            
        #     frame_len = reader.get_length()
        #     print('{} has {} frames'.format(video, frame_len))

        #     feature = []
        #     from time import time
        #     batch_split = np.array_split(np.arange(frame_len), frame_len//batch_size)
        #     for no, indices in enumerate(batch_split):
        #         print('{}/{}'.format(no+1, len(batch_split)))
        #         frames = [resnet_transform(reader.get_next_data()) for i in indices]
        #         frames = torch.stack(frames).to(device=devices)
        #         with torch.no_grad():
        #             tic = time()
        #             _, feat = net(frames)
        #             toc=time()
        #             print('cost {:.4f}'.format(toc-tic))
        #             feat=feat.to(device=devices)
        #             feature.append(feat)
        #             torch.cuda.empty_cache()

        #             if use_gpu:
        #                 print(torch.cuda.memory_allocated()/2**20, 'MB')
        #                 print(torch.cuda.memory_cached()/2**20, 'MB')
        #                 print()                    
        #     feature = torch.cat(feature)

        #     with h5py.File(h5file_name) as f:
        #         d = f.create_dataset('feat', data=feature.numpy())
        #         d.attrs['vname'] = video[:-4]
            
        #     reader.close()
        #     print(video_name, 'finished')


        # method 3: need opencv preprocess
        frames_root_dir = '/slwork/jun/vsum_project/Datasets/VSUMM_frames'
        for ty in data_type:
            raw_frames_dir = os.path.join(frames_root_dir, ty)
            
            for work_dir in sorted(os.listdir(raw_frames_dir), key=lambda s:int(s[1:])):
                h5file_name = '{}.hdf5'.format(os.path.join(feature_output_dir, work_dir))
                print('processing ', h5file_name)
                if os.path.exists(h5file_name):
                    print('existed, skip')
                    continue
                
                frames = []
                wd = os.path.join(frames_root_dir, ty, work_dir)
                frames_name = sorted(os.listdir(wd), key=lambda s:int(s[:-4]))
                for name in frames_name:
                    img = imread(os.path.join(wd, name))
                    frames.append(resnet_transform(img))
                frames = torch.stack(frames)
                print(work_dir, frames.shape)

                feature = []
                frame_len = frames.shape[0]
                batch_split = np.array_split(np.arange(frame_len), frame_len//batch_size)
                for no, indices in enumerate(batch_split):
                    print('{}/{}'.format(no+1, len(batch_split)))
                    with torch.no_grad():
                        tic = time()
                        _, feat = net(frames[indices].to(device=devices))
                        toc=time()
                        print('cost {:.4f}'.format(toc-tic))
                        feat=feat.to(device='cpu')
                        feature.append(feat)

                        if use_gpu:
                            print(torch.cuda.memory_allocated()/2**20, 'MB')
                            print(torch.cuda.memory_cached()/2**20, 'MB')
                            torch.cuda.empty_cache()
                            print()                    
                feature = torch.cat(feature)

                with h5py.File(h5file_name) as f:
                    d = f.create_dataset('feat', data=feature.numpy())
                    d.attrs['vname'] = work_dir



if __name__ == '__main__':
    main()

