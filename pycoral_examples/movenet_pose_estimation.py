# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to estimate a single human pose with Edge TPU MoveNet.

To run this code, you must attach an Edge TPU to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

For more details about MoveNet and its best practices, please see
https://www.tensorflow.org/hub/tutorials/movenet

Example usage:
```
python3 movenet_pose_estimation.py \
  --model models/pose/movenet_single_pose_lightning_ptq_edgetpu.tflite
```
"""

import argparse
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
from flask import Flask, Response

app = Flask(__name__)

_NUM_KEYPOINTS = 17
interpreter = None
cap = None
input_size = None

def gen_frames():
  global cap, interpreter, input_size
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # Convert BGR (OpenCV) to RGB (PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # Resize for model input
    resized_img = pil_img.resize(input_size, Image.LANCZOS)
    common.set_input(interpreter, resized_img)

    # Run inference
    interpreter.invoke()

    # Get pose keypoints
    pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)

    # Draw keypoints on original frame
    height, width, _ = frame.shape
    for i in range(0, _NUM_KEYPOINTS):
      y = int(pose[i][0] * height)
      x = int(pose[i][1] * width)
      confidence = pose[i][2]

      # Draw keypoint if confidence is high enough
      if confidence > 0.3:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "<body><h1>MoveNet Live Feed</h1><img src='/video_feed' width='640' height='480' /></body>"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



def main():
  global interpreter, cap, input_size
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '--camera_idx',
      type=int,
      default=0,
      help='Camera device index.')
  args = parser.parse_args()

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()
  input_size = common.input_size(interpreter)

  # Open webcam
  cap = cv2.VideoCapture(args.camera_idx)
  if not cap.isOpened():
    print(f"Error: Could not open camera at index {args.camera_idx}")
    return

  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
  print(f"Camera opened: {width}x{height}")
  print("Starting Flask server at http://0.0.0.0:5000")

  app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
  main()
