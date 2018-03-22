import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
	"""Filters YOLO boxes by thresholding on object and class confidence."""
	
	# Step 1: Compute box scores
	box_scores = box_confidence*box_class_probs
	
	# Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
	box_classes = K.argmax(box_scores,-1)
	box_class_scores = K.max(box_scores,-1)
	
	# Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
	filtering_mask = box_class_scores>threshold
	
	# Step 4: Apply the mask to scores, boxes and classes
	scores = tf.boolean_mask(box_class_scores,filtering_mask)
	boxes = tf.boolean_mask(boxes,filtering_mask)
	classes = tf.boolean_mask(box_classes,filtering_mask)
		
	return scores, boxes, classes



def iou(box1, box2):
	"""Implement the intersection over union (IoU) between box1 and box2 """

	# Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
	xi1 = max(box1[0],box2[0])
	yi1 = max(box1[1],box2[1])
	xi2 = min(box1[2],box2[2])
	yi2 = min(box1[3],box2[3])
	inter_area = (yi2-yi1)*(xi2-xi1)

	# Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
	box1_area = (box1[3]-box1[1])*(box1[2]-box1[0])
	box2_area = (box2[3]-box2[1])*(box2[2]-box2[0])
	union_area = box1_area+box2_area-inter_area

	# compute the IoU
	iou = inter_area/union_area

	return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
	""" Applies Non-max suppression (NMS) to set of boxes """
	
	max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
	K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
	
	# Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
	nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold)
	
	# Use K.gather() to select only nms_indices from scores, boxes and classes
	scores = K.gather(scores,nms_indices)
	boxes = K.gather(boxes,nms_indices)
	classes = K.gather(classes,nms_indices)
	
	return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
	""" Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes. """
		
	# Retrieve outputs of the YOLO model (≈1 line)
	box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

	# Convert boxes to be ready for filtering functions 
	boxes = yolo_boxes_to_corners(box_xy, box_wh)

	# Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
	
	# Scale boxes back to original image shape.
	boxes = scale_boxes(boxes, image_shape)

	# Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
	scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
	
	return scores, boxes, classes


sess = K.get_session()


class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)  

yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)




def predict(sess, image_file):
	""" Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions. """

	# Preprocess your image
	image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

	# Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
	out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})

	# Generate colors for drawing bounding boxes.
	colors = generate_colors(class_names)
	
	# Draw bounding boxes on the image file
	draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
	
	# Save the predicted bounding box on the image
	image.save(os.path.join("out", image_file), quality=90)
	
	# Display the results in the notebook
	output_image = scipy.misc.imread(os.path.join("out", image_file))
	
	return out_scores, out_boxes, out_classes


#Iterating through the test dataset and applying detection + localization to the objects
directory = os.fsencode("images")

for file in tqdm(os.listdir(directory)):
	filename = os.fsdecode(file)
	if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith("jpeg"): 
		out_scores, out_boxes, out_classes = predict(sess, filename)



