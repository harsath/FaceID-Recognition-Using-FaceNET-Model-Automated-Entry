from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc  
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import csv
from datetime import datetime


input_video = "source.mov"  # Real time Video detection
modeldir = './model/20170511-185253.pb'
# Path to Classifier file
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = "./train_img"


def unique_names(lists):
    ar_uniq = []
    for i in lists:
        if not i in ar_uniq:
            ar_uniq.append(i)
    return ar_uniq


global lists_res
lists_res = []
text = []

img_dir_name = "detec_faces"
if not os.path.exists(img_dir_name):
    os.mkdir(img_dir_name)

# <EXT for each Block>
with tf.Graph().as_default():
    #Using the 99% of the CUDA Cores to perform the Calculations
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 10
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        HumanNames = os.listdir(train_img)
        HumanNames.sort()

        print('Loading Model')
        facenet.load_model(modeldir)  # PreTrained Model Loaded
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        # Placeholder for the Mappings of the Descriet Faces in Space in form of n Dimentional Tensor(Real Numbers)
        # The Classifier Takes advantages of the Patten in the Vectors
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename_exp = os.path.expanduser(classifier_filename)
        # Loading the Classifier into the Memory
        print("Reading the Classifier Binary")
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        video_capture = cv2.VideoCapture(0)  # Accessing the Camera{WEB:1}
        c = 0
        fps_capture = video_capture.get(cv2.CAP_PROP_FPS)
        print('Start Recognition')
        prevTime = 0
        counter = 1

        while True:
            ret, frame = video_capture.read()

            # resize frame (optional)
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            curTime = time.time()+1    # calc fps
            timeF = frame_interval  # 3 Frames Between the Intrevel

            if (c % timeF == 0): #True for ever since not Updating
                find_results = []
                #If Gray scale, turning it into a Color image
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                # Getting the Boundry Boxes for the Face(After Window Detection)
                bounding_boxes, _ = detect_face.detect_face(
                    frame, minsize, pnet, rnet, onet, threshold, factor)
                # Getting Number Of Detected Faces In A Frame
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)
                print(f"FPS : {fps_capture}")

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('Face is very close!')
                            continue

                        cropped.append(
                            frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        try:
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(
                                cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            # Appending the Scaled Image into the Tensorflow Placeholder
                            scaled_reshape.append(
                                scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                            feed_dict = {
                                images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(
                                embeddings, feed_dict=feed_dict)
                            # Printing the Prediction on All Classes
                            predictions = model.predict_proba(emb_array)
                            print(predictions)
                            # Getting the Highest of the Prediction(Argmax Function)
                            # This returns the Indeces of the Array(Heighest)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(
                                len(best_class_indices)), best_class_indices]
                            # print("predictions")
                            print(best_class_indices, ' with accuracy ',
                                  best_class_probabilities)

                            # print(best_class_probabilities)
                            # Setting up the Threshold Value before showing the Text
                            if best_class_probabilities > 0.55:
                                # boxing face
                                cv2.rectangle(
                                    frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)

                                # plot result idx under box
                                # Getting the Co-Ordinates for the X,Y Axis
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                print('Result Indices: ',
                                      best_class_indices[0])
                                # This array of All Trained Classes
                                print(HumanNames)
                                for H_i in HumanNames:
                                    # If Predicted and Loop matches:<Block>
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        # Putting the Predicted Class in the Boundry Box
                                        cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 255), thickness=1, lineType=2)
                                        lists_res.append(result_names)
                                        cv2.imwrite(f"{img_dir_name}/{result_names}.jpg", frame)
                                        counter += 1
                        except Exception as e:
                            pass

                else:
                    print('Alignment Failure')
            # c+=1
            cv2.imshow('Video', frame)
            x = datetime.now()
            csv_date_format = f"{x.day}-{x.month}-{x.year}"
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # with open('names.txt',"w") as f:
                # 	s = unique_names(lists_res)

                print(set(lists_res))
                li = list(set(lists_res))

                with open(f"{csv_date_format}.csv", "a", newline="") as f:
                    fieldnames = ["Names"]
                    writer = csv.DictWriter(f, fieldnames)
                    writer.writeheader()
                    for i in li:
                        writer.writerow({'Names': i})

                break

        video_capture.release()
        cv2.destroyAllWindows()
