#
# Copyright 2023 Alexander Rose. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import logging
import time
import os
import io
import re
import glob
import base64
import warnings
import timeit
from typing import Any, Sequence

import numpy as np
from PIL import Image
import cv2 as cv

# Ignore deprecation and future warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

# Capture return value with the timeit module
timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

# Read environment variables
MODEL = os.getenv('MODEL', None)
CONFIG = os.getenv('CONFIG', None)
LAMBDA_TASK_ROOT = os.environ.get('LAMBDA_TASK_ROOT')
CLASSES = os.getenv('CLASSES', None)
FORWARD_PASS = os.getenv('FORWARD_PASS', '').lower() == 'true'
SILENT_RUN = os.getenv('SILENT_RUN', '').lower() == 'true'
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
NMS_THRESHOLD = float(os.getenv('NMS_THRESHOLD', 0.5))
SCALE_FACTOR = os.getenv('SCALE_FACTOR', 1.0)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'notset').lower()

# Configure logging
log_levels = dict(
    notset=logging.NOTSET,
    debug=logging.DEBUG,
    info=logging.INFO,
    warning=logging.WARNING,
    error=logging.ERROR,
    critical=logging.CRITICAL
)
log_level = log_levels.get(LOG_LEVEL, logging.NOTSET)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(log_level)
    logger = logging.getLogger()
else:
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

if MODEL is None or CONFIG is None:
    """
    Check if MODEL and CONFIG are already defined. If not, perform the following steps:
    
    1. Define patterns to match desired file extensions.
    2. Get all files in the current directory and subdirectories.
    3. Filter files based on the file extensions.
    4. Find the first matching model file.
    5. Check the file extension of the MODEL and find the corresponding configuration file.
    6. Check if CONFIG is None, unless MODEL ends with .t7, .net, or .onnx.
    """
    model_pattern = r"\.(caffemodel|pb|t7|net|weights|bin|onnx)$"
    config_pattern = r"\.(prototxt|pbtxt|cfg|xml)$"
    files = [file for file in glob.glob(f"{LAMBDA_TASK_ROOT}/**/*", recursive=True)]
    config_files = [file for file in files if re.search(config_pattern, file)]
    model_files = [file for file in files if re.search(model_pattern, file)]
    MODEL = MODEL or next((c for c in model_files), None)
    if MODEL is None:
        raise ValueError("MODEL cannot be None")
    elif MODEL.endswith('.caffemodel'):
        deploy_prototxt = next((c for c in config_files if c.endswith('deploy.prototxt')), None)
        CONFIG = CONFIG or deploy_prototxt or next((c for c in config_files if c.endswith('.prototxt')), None)
    elif MODEL.endswith('.pb'):
        CONFIG = CONFIG or next((c for c in config_files if c.endswith('.pbtxt')), None)
    elif MODEL.endswith('.weights'):
        CONFIG = CONFIG or next((c for c in config_files if c.endswith('.cfg')), None)
        if "yolo" in MODEL.lower():
            SCALE_FACTOR = os.getenv('SCALE_FACTOR', (1.0 / 255))
    elif MODEL.endswith('.bin'):
        CONFIG = CONFIG or next((c for c in config_files if c.endswith('.xml')), None)
    if CONFIG is None and not (
            MODEL.endswith('.t7') or MODEL.endswith('.net') or MODEL.endswith('.onnx') or MODEL.endswith('.pb')):
        raise ValueError("CONFIG cannot be None")


def forward(model: str, config: str, img: np.ndarray) -> Sequence[Any]:
    """
    Forward pass an image through a model.

    :param model: Path to model file.
    :type model: str
    :param config: Path to configuration file.
    :type config: str
    :param img: Input image for forward pass.
    :type img: numpy.ndarray
    :return: Output obtained from the forward pass.
    :rtype: Sequence[Any]
    """
    height, width, _ = img.shape
    net = cv.dnn.readNet(model, config)
    net.setInput(cv.dnn.blobFromImage(img, scalefactor=SCALE_FACTOR, size=(width, height), swapRB=True, crop=False))
    return net.forward(net.getUnconnectedOutLayersNames())


def detect(model: str, config: str, img: np.ndarray) -> tuple[np.ndarray]:
    """
    Perform object detection on an image using the specified model and configuration.

    :param model: Path to the model file.
    :type model: str
    :param config: Path to the configuration file.
    :type config: str
    :param img: Image on which to perform object detection.
    :type img: numpy.ndarray
    :return: Detected objects.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    height, width, _ = img.shape
    net = cv.dnn.readNet(model, config)
    detection_model = cv.dnn_DetectionModel(net)
    detection_model.setInputSize(width, height)
    detection_model.setInputScale(SCALE_FACTOR)
    detection_model.setInputSwapRB(True)
    return detection_model.detect(img, **dict(
        confThreshold=CONFIDENCE_THRESHOLD,
        nmsThreshold=NMS_THRESHOLD
    ))


def label(res: tuple) -> list[dict]:
    """
    Generate labeled objects from detection results.

    :param res: Detection results as a tuple containing class IDs, confidences, and bounding boxes.
    :return: Labeled objects as a list of dictionaries with ID, confidence, and bounding box coordinates.
    """
    try:
        class_ids, confidences, boxes = res
        return [
            dict(
                id=res[0],
                confidence=res[1],
                left=int(res[2][0]),
                top=int(res[2][1]),
                right=int(res[2][0] + res[2][2]),
                bottom=int(res[2][1] + res[2][3]),
            ) for res in list(zip(class_ids.tolist(), confidences.tolist(), boxes.tolist()))
        ]
    except AttributeError:
        return []


def infer(data: bytes, model_path: str, config_path: str, names_path: str = None, **kwargs) -> dict:
    """
    Performs object detection and returns the inference computation time and labeled predictions.

    :param data: Input data as bytes representing an image.
    :type data: bytes
    :param model_path: Path to the model file.
    :type model_path: str | os.PathLike
    :param config_path: Path to the configuration file.
    :type config_path: str | os.PathLike
    :param names_path: Optional path to the file containing class names.
    :type names_path: str | os.PathLike, optional
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict, optional
    Keyword Arguments:
        * forward (bool): Optional flag indicating whether to perform a forward pass and return the computation time and response.
    :return: A dictionary containing inference statistics and labeled objects.
    :rtype: dict
    :warns: FileNotFoundError: If the `names_path` file is not found.
    """
    with Image.open(io.BytesIO(data)) as pil_img:
        img = cv.cvtColor(np.array(pil_img), cv.COLOR_BGR2RGB)

    if kwargs.get('forward', False):
        t0 = time.time()
        l, resp = timeit.timeit(lambda: forward(model_path, config_path, img), number=1)
        t1 = time.time()
        results = tuple([ctx.tolist() for ctx in resp])
        return dict(latency=l, start_time=t0, end_time=t1, results=results)

    t0 = time.time()
    l, resp = timeit.timeit(lambda: detect(model_path, config_path, img), number=1)
    t1 = time.time()
    results = label(resp)

    try:
        if names_path:
            with open(names_path, 'r') as fp:
                names = fp.read().rstrip("\n").split("\n")
                for lab in results:
                    lab.update(dict(id=names[lab.get("id")]))
    except FileNotFoundError as e:
        logger.warning(e)
        pass

    return dict(latency=l, start_time=t0, end_time=t1, results=results)


def lambda_handler(event, context):
    """
    Lambda handler for inference.
    """
    t0 = time.time()
    st = time.perf_counter_ns()

    logger.debug("model: %s", MODEL)
    logger.debug("config: %s", CONFIG)

    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": "{}"
    }

    try:
        body = event['body']
        image_bytes = body.encode('utf-8')
        img_b64dec = base64.b64decode(image_bytes)

        resp = infer(img_b64dec, MODEL, CONFIG, CLASSES, forward=FORWARD_PASS)

        detection_start_time = resp.get('start_time')
        detection_end_time = resp.get('end_time')
        detection_latency = resp.get('latency')

        if SILENT_RUN:
            detection_results = []
        else:
            detection_results = resp.get('results')

        latency = (time.perf_counter_ns() - st) / 10 ** 9
        t1 = time.time()

        response['body'] = json.dumps(
            dict(
                Model=os.path.basename(MODEL),
                Config=os.path.basename(CONFIG),
                ForwardPass=FORWARD_PASS,
                SilentRun=SILENT_RUN,
                StartTime=t0,
                EndTime=t1,
                Latency=latency,
                DetectionStartTime=detection_start_time,
                DetectionEndTime=detection_end_time,
                DetectionLatency=round(detection_latency, 9),
                Results=detection_results,
            )
        )

    except Exception as e:
        logger.error("error: %s", e)
        response = {
            "statusCode": 500
        }
    finally:
        logger.info(response)
        return response
