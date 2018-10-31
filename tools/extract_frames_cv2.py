import numpy as np
import cv2
import os

root_dir = 'VSUMM'
type_dir = 'new_database'
frames_root_dir = 'VSUMM_frames'
image_type = 'jpg'

def main():
    for video_name in os.listdir(os.path.join(root_dir, type_dir)):
        print('extracting {}'.format(video_name))
        video_base_name = video_name[:-4]
        frames_dir = os.path.join(frames_root_dir, type_dir, video_base_name)
        if os.path.exists(frames_dir):
            print('existed, skip')
            continue
        
        os.makedirs(frames_dir)

        frames = extract_frames(os.path.join(root_dir, type_dir, video_name))
        for i, frame in enumerate(frames):
            frame_name = os.path.join(frames_dir, '{}.{}'.format(i,image_type))
            cv2.imwrite(frame_name, frame)
        
        print('{} extraction completed.'.format(frames_dir))
   

def extract_frames(video_name, display=False):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_name)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Read until video is completed
    frames = []
    while(cap.isOpened()):
        # Capture frame-by-frame`
        ret, frame = cap.read()
            
        if ret == True:
            frames.append(frame)
            if display:
                # Display the resulting frame
                cv2.imshow('Frame',frame)
            
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        
        # Break the loop
        else: 
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    if display: cv2.destroyAllWindows()
    
    return np.array(frames)

if __name__ == '__main__':
    main()
