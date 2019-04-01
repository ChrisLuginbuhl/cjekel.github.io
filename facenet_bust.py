#Uses David Sandberg's Facenet implementation, using an adaptation of Charles Jekel's script for cropping
# To run face detection:
# Do NOT source activate tensorflow_p36impor
# Clone https://github.com/davidsandberg/facenet
# Run pip install -r requirements.txt (note that this changes tensor flow version)
# Pip install align
# Pip install gensim
# export PYTHONPATH=$PYTHONPATH:~/facenet/src
# Note the tilda above!
# Clone or upload facenet_bust.py
# Put images in subfolder /early_renaissance
# Mkdir early_renaissance/outputs
# Python facenet_bust.py


#   import facenet libraires
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from scipy import misc
import tensorflow as tf
import os
import align.detect_face
from PIL import Image

#  import other libraries
import cv2
#import matplotlib.pyplot as plt

#   setup facenet parameters
gpu_memory_fraction = 1.0
minsize = 50 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

wExpansion = 1.25   # fraction of face bounding box width to add to width. int or float is fine
hExpansionUp = 0    # fraction of face boudning box height to add to top
hExpansionDown = 0.9


#   fetch images
#image_dir = 'presidents/'
image_dir = 'early_renaissance/'

#   create a list of your images
filenames = []
images = os.listdir(image_dir)
print("Num images: ", len(images))
for filename in images:
    if filename.endswith( ('.jpeg', '.jpg', '.png', '.gif') ): # whatever file types you're using...
        filenames.append(filename)
filenames.sort()

print(filenames)
print("Num filenames: ", len(filenames))

#   Start code from facenet/src/compare.py
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
        log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(
            sess, None)
#   end code from facenet/src/compare.py

    for i in filenames:
        input_path = image_dir + i
        img = misc.imread(os.path.expanduser(input_path))
        f, e = os.path.splitext(i)
        #   run detect_face from the facenet library
        bounding_boxes, _ = align.detect_face.detect_face(
                img, minsize, pnet,
                rnet, onet, threshold, factor)
        # make a separate file for each detected face
        k = 0
        for (x1, y1, x2, y2, acc) in bounding_boxes:
            print(i, int(x1), int(x2), int(y1), int(y2))
            bustX1 = int(x1 - int((x2-x1) * wExpansion / 2))  # calculate larger bounding boxes for bust
            bustX2 = int(x2 + int((x2 - x1) * wExpansion / 2))
            bustY1 = int(y1 - (y2 - y1) * hExpansionUp)
            bustY2 = int(y2 + (y2 - y1) * hExpansionDown)
            print(i, bustX1, bustX2, bustY1, bustY2)
            w = bustX2 - bustX1
            h = bustY2 - bustY1
            im = Image.open(os.path.expanduser(image_dir + i))
            new_im = Image.new("RGB", (h, h))  # make a square image, assuming height is always the largest dimension
            box = (bustX1, bustY1, bustX2, bustY2)
            new_im.paste(im.crop(box), (int((h-w)/2), 0))  # 2nd parameter is where to paste in x,y - centre the image l-R
            new_im.save('early_renaissance/outputs/' + f + '_' + str(k) + '.jpg', 'JPEG', quality=70)
            k += 1
