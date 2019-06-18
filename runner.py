import os

dir_name_v = "video"
s = os.listdir(dir_name_v)
if len(s) >0:
	for i,file in enumerate(s):
		names = file.split(".")[0]
		os.mkdir(names)
		os.system(f"python3 video2frames.py video/{s[i]} {names}")
		print("Video Converted into Frames")
		os.system(f"mv {names} train_img")
		print("Moved Successfully")
		os.system("python3 data_preprocess.py")
		print("Training the Model")
		os.system("python3 train_main.py")







