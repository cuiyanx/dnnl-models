from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
from matplotlib import pyplot
import pylab
import os
from caffe2.python import core, workspace, models, brew, model_helper
import google.protobuf.text_format as ptxt
import operator
print("Required modules imported.")

CAFFE_MODELS = "../model"
IMAGE_LOCATION = "./ILSVRC2012_val_00033000.JPEG"
MODEL = ["squeezenet", "init_net.pb", "predict_net.pb", 224]
codes = "./synset.txt"
MEAN_FILE = "./ilsvrc_2012_mean.npy"
print("Config set!")

CAFFE_MODELS = os.path.expanduser(CAFFE_MODELS)

mean = np.load(MEAN_FILE).mean(1).mean(1)
# mean = [128, 128, 128]
# mean = [104, 117, 123]
# mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
print("mean was set to: ", mean)
print("std was set to: ", std)

INPUT_IMAGE_SIZE = MODEL[3]
INIT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[1])
PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[2])
if not os.path.exists(INIT_NET):
    print("WARNING: " + INIT_NET + " not found!")
else:
    if not os.path.exists(PREDICT_NET):
        print("WARNING: " + PREDICT_NET + " not found!")
    else:
        print("All needed files found!")

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    # Get original aspect ratio
    aspect = img.shape[1]/float(img.shape[0])
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    return imgScaled

img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
print("Original Image Shape: " , img.shape)
pyplot.figure()
pyplot.imshow(img)
pyplot.title('Original image')
# pyplot.show()

img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print("Image Shape after rescaling: " , img.shape)
pyplot.figure()
pyplot.imshow(img)
pyplot.title('Rescaled image')
# pyplot.show()

img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print("Image Shape after cropping: " , img.shape)
pyplot.figure()
pyplot.imshow(img)
pyplot.title('Center Cropped')
# pyplot.show()

# switch to CHW (HWC --> CHW)
img = img.swapaxes(1, 2).swapaxes(0, 1)
print("CHW Image Shape: " , img.shape)

pyplot.figure()
for i in range(3):
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(img[i], cmap=pyplot.cm.gray)
    pyplot.title('RGB channel %d' % (i+1))
    # pyplot.show()

# switch to BGR (RGB --> BGR)
img = img[(2, 1, 0), :, :]

pyplot.figure()
for i in range(3):
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(img[i], cmap=pyplot.cm.gray)
    pyplot.title('BGR channel %d' % (i+1))
    # pyplot.show()

# remove mean for better results
img = img * 255 - mean[:, np.newaxis, np.newaxis]
# img = img * 255 - np.array(mean).reshape(3, 1, 1)

# add batch size axis
img = img[np.newaxis, :, :, :].astype(np.float32)

print("NCHW image (ready to be used as input): ", img.shape)

workspace.ResetWorkspace()
device_opts = core.DeviceOption(caffe2_pb2.CPU, 0)
# device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)
# device_opts = core.DeviceOption(caffe2_pb2.IDEEP, 0)

with open(INIT_NET, "rb") as f:
    # init_def = ptxt.Parse(f.read(), caffe2_pb2.NetDef())
    init_def = caffe2_pb2.NetDef()
    init_def.ParseFromString(f.read())
    init_def.device_option.CopyFrom(device_opts)
with open(PREDICT_NET, "rb") as f:
    # predict_def = ptxt.Parse(f.read(), caffe2_pb2.NetDef())
    predict_def = caffe2_pb2.NetDef()
    predict_def.ParseFromString(f.read())
    predict_def.device_option.CopyFrom(device_opts)

workspace.RunNetOnce(init_def)
workspace.FeedBlob(predict_def.op[0].input[0], img, device_opts)
workspace.CreateNet(predict_def)
workspace.RunNet(predict_def.name, 1)
results = workspace.FetchBlob(predict_def.op[-1].output[0])

'''
p = workspace.Predictor(init_net, predict_net)
results = p.run({'data': img})
'''

results = np.asarray(results)
print("results shape: ", results.shape)

preds = np.squeeze(results)
curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))
print("Prediction: ", curr_pred)
print("Confidence: ", curr_conf)

# the rest of this is digging through the results
results = np.delete(results, 1)
index = 0
highest = 0
arr = np.empty((0,2), dtype=object)
arr[:,0] = int(10)
arr[:,1:] = float(10)
for i, r in enumerate(results):
    # imagenet index begins with 1!
    i=i+1
    arr = np.append(arr, np.array([[i,r]]), axis=0)
    if (r > highest):
        highest = r
        index = i

# top N results
N = 5
topN = sorted(arr, key=lambda x: x[1], reverse=True)[:N]
print("Raw top {} results: {}".format(N,topN))

# Isolate the indexes of the top-N most likely classes
topN_inds = [int(x[0]) for x in topN]
print("Top {} classes in order: {}".format(N,topN_inds))

# Now we can grab the code list and create a class Look Up Table
with open(codes) as f:
    response = f.readlines()

    class_LUT = []
    for (line_num, line_text) in enumerate(response):
        line_text = str(line_text)[9:].replace("\n", "")
        line_num = line_num + 1
        class_LUT.append(line_text.split(",")[0][1:])

# For each of the top-N results, associate the integer result with an actual class
for n in topN:
    print("Model predicts '{}' with {}% confidence".format(class_LUT[int(n[0])],float("{0:.2f}".format(n[1]*100))))
