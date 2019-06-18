import cv2
import os
import argparse

#CLI
args = argparse.ArgumentParser(description="Video to Frames")
args.add_argument("video_file",help="Video File", type=str)
args.add_argument("dir_name", help="Dir Name", type=str)

argmen = args.parse_args()
video_name = argmen.video_file
dir_name = argmen.dir_name

vidcap = cv2.VideoCapture(video_name)
success,out = vidcap.read()
count = 0


if not os.path.isdir(dir_name):
	os.mkdir(dir_name)

while count <= 150:

  out=cv2.transpose(out)
  #out=cv2.flip(out,flipCode=0)
  cv2.imwrite(f"{dir_name}/frame{count}d.jpg" , out)     # save frame as JPEG file
  success,out = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
