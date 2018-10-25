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
resnet_input_shape = [224, 224,]
batch_size = 1

def main():
    net = ResNetFeature(feature='resnet101')
    resnet_transform = transforms.Compose([
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
            feature = []
            
            frame_no = reader.get_length()
            print('{} has {} frames'.format(video, frame_no))
            index_split = np.array_split(np.arange(frame_no), frame_no//batch_size)

            for i, s in enumerate(index_split):
                frames = [reader.get_next_data() for i in s]
                frames = np.array(frames)
                
                for j, f in enumerate(frames):
                    resize_frames[j] = resize(f, resnet_input_shape)
                convert_np_2_tensor = lambda f: torch.from_numpy(f.astype(np.float32)).permute(0, 3, 1, 2).cuda(0)

                res5c, pool5 = net(resnet_transform(convert_np_2_tensor(resize_frames)))

                del resize_frames

                print(pool5.size())
                feature.append(pool5.cpu())

                del res5c, pool5
                print('batch %i'%i)
            features = torch.cat(feature)

            with h5py.File(h5file_name) as f:
                d = f.create_dataset('feat', data=features.numpy())
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
