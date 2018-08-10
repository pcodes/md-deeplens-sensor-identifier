#
# Copyright 2010-2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#

# greengrassHelloWorld.py
# Demonstrates a simple publish to a topic using Greengrass core sdk
# This lambda function will retrieve underlying platform information and send
# a hello world message along with the platform information to the topic 'hello/world'
# The function will sleep for five seconds, then repeat.  Since the function is
# long-lived it will run forever when deployed to a Greengrass core.  The handler
# will NOT be invoked in our example since the we are executing an infinite loop.

import os
import sys
from threading import Thread, Event
import tensorflow as tf
import awscam
import cv2
import numpy as np
import time

ret, frame = awscam.getLastFrame()
ret, jpeg = cv2.imencode('.jpg', frame)
Write_To_FIFO = True

MODEL_NAME = 'tensorflow-model'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'graph.pb')
NUM_CLASSES = 2
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'labels.txt')

detection_graph = tf.Graph()
graph_def = tf.GraphDef()
with open(PATH_TO_CKPT, "rb") as f:
	graph_def.ParseFromString(f.read())
with detection_graph.as_default():
	tf.import_graph_def(graph_def)

labels = []
proto_as_ascii_lines = tf.gfile.GFile("labels.txt").readlines()
for l in proto_as_ascii_lines:
	labels.append(l.rstrip())

class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """
    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()

def greengrass_infinite_infer_run():
	try:
    	#input_width = 300
       	#input_height = 300
    	#prob_thresh = 0.65
		results_thread = LocalDisplay('720p')
		results_thread.start()

		#ret, frame = awscam.getLastFrame()
		#if ret == False:
		#	raise Exception("Failed to get frame from stream")

    	#yscale = float(frame.shape[0] / input_height)
    	#xscale = float(frame.shape[1] / input_width)

		#input_name = "file_reader"
		#output_name = "normalized"
		with tf.Session(graph=detection_graph) as sess:
			while True:
				ret, frame = awscam.getLastFrame()
				
				if ret == False:
					print "Did not get video"
					raise Excepion("Failed to get frame from stream")

				#add image resizing here
				frame_expanded = tf.expand_dims(frame, axis=0)
				frame_resized = tf.image.resize_bilinear(frame_expanded, [299, 299])
				normalized = tf.divide(tf.subtract(frame_resized, [0]), [255])
				new_sess = tf.Session()
				image_tensor = new_sess.run(normalized)
				input_name = "import/Placeholder"
				output_name = "import/final_result"
				input_operation = detection_graph.get_operation_by_name(input_name)
				output_operation = detection_graph.get_operation_by_name(output_name)

				results = sess.run(output_operation.outputs[0], {
					input_operation.outputs[0]: image_tensor
				})

				results = np.squeeze(results)
				
				top_k = results.argsort()[-5:][::-1]
				top_item = labels[top_k[0]]

				#log_path = "/tmp/log.txt"
				#log_file = open(log_path, "a+")

				for i in top_k:
					print labels[i], results[i]
				print "---"
				
				if results[top_k[0]] > 0.7:
					cv2.putText(frame, "Object: " + top_item, (640, 1300), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 5)
				results_thread.set_frame_data(frame)


			#global jpeg
			#ret, jpeg = cv2.imencode('.jpg', frame)

	except Exception as e:
		print e

greengrass_infinite_infer_run()

def function_handler(event, context):
    return
