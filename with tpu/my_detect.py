import os
import argparse
import numpy as np
import sys
import time
import importlib.util
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import matplotlib.pyplot as plt
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
@app.route("/")
def index() :
    return render_template('index.html')

modelPath = os.path.join(os.getcwd(), "detect_24000.tflite")
labelPath = os.path.join(os.getcwd(), "label.txt")

def load_label(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}
        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}

def get_frame():
    cap = cv2.VideoCapture(0)
    interpreter = Interpreter(modelPath, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width = input_details[0]['shape'][2]
    height = input_details[0]['shape'][1]
    labels = load_label(labelPath)
    image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_counter = 0
    start = time.time()
    while True :
        _, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        
        for i in range(len(scores)):
            if ((scores[i] > 0.2) and (scores[i] <= 1.0)):
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * image_height)))
                xmin = int(max(1, (boxes[i][1] * image_width)))
                ymax = int(min(image_height, (boxes[i][2] * image_height)))
                xmax = int(min(image_width, (boxes[i][3] * image_width)))

                cv2.rectangle(frame, (xmin, ymin),
                              (xmax, ymax), (10, 255, 0), 4)

                                # Draw label
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i]*100))
                print(object_name, int(scores[i]*100))
                print("x : ({}, {}) y : ({}, {})".format(xmin, xmax, ymin, ymax))
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Make sure not to draw label too close to top of window
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (
                    xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tostring()
        
        yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\n' + stringData + b'\r\n')

    del(cap)

def get_frame_o():
    interpreter = Interpreter(modelPath)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width = input_details[0]['shape'][2]
    height = input_details[0]['shape'][1]

    labels = load_label(labelPath)

        # Capturing the video.
    cap = cv2.VideoCapture(0)
    image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_counter = 0
    start = time.time()
    while(True):
        frame_counter += 1
        # Acquire frame and resize to expected shape [1xHxWx3]
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # set frame as input tensors
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # perform inference
        interpreter.invoke()

        # Get output tensor
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > 0.2) and (scores[i] <= 1.0)):
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * image_height)))
                xmin = int(max(1, (boxes[i][1] * image_width)))
                ymax = int(min(image_height, (boxes[i][2] * image_height)))
                xmax = int(min(image_width, (boxes[i][3] * image_width)))

                cv2.rectangle(frame, (xmin, ymin),
                              (xmax, ymax), (10, 255, 0), 4)

                                # Draw label
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i]*100))
                print(label)
                print("x : ({} , {}), y : ({}, {})".format(xmin, xmax, ymin, ymax))
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Make sure not to draw label too close to top of window
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (
                    xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
        if time.time() - start >= 1:
            print('fps:', frame_counter)
            frame_counter = 0
            start = time.time()
        
        yield(b'--frame\r\n')

    # Clean up
    del(cap)

@app.route('/calc')
def calc() :
    return Response(get_frame(),
            mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, threaded=True)
