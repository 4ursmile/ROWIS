import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import glob
import multiprocessing as mp
import os
import time
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from sparseinst import VisualizationDemo, add_sparse_inst_config

st.set_page_config(page_title='SparseInst Demo', page_icon=':smiley:')
st.write('This is a demo of SparseInst, a sparse instance segmentation model. It is based on Detectron2 and Sparse R-CNN.')
st.write('The model is trained on COCO dataset. Use R50D_CDN backbone')

# input image and confidence threshold
with st.form(key='my_form'):
    image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    confidence_threshold = st.slider('Confidence threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    submit_button = st.form_submit_button(label='Submit')
# process image
if submit_button:
    # constants
    WINDOW_NAME = "COCO detections"

    def setup_cfg(args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_sparse_inst_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        cfg.MODEL.WEIGHTS = 'models/sparseinst_r50vd_dcn_giam_aug.pth'
        cfg.freeze()
        return cfg

    def get_parser():
        parser = argparse.ArgumentParser(
            description="Detectron2 demo for builtin models")
        parser.add_argument(
            "--config-file",
            default="configs/sparse_inst_r50vd_dcn_giam_aug.yaml",
            metavar="FILE",
            help="path to config file",
        )
        parser.add_argument(
            "--output",
            default="result",
            help="A file or directory to save output visualizations. "
            "If not given, will show output in an OpenCV window.",
        )

        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Minimum score for instance predictions to be shown",
        )
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )
        return parser

    args = get_parser().parse_args(['--config-file', 'configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml', '--input', 'input.jpg', '--confidence-threshold', str(confidence_threshold)])

    logger = setup_logger(name="detectron2")
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if image is not None:
        # use PIL, to be consistent with evaluation
        img = read_image(image, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img, confidence_threshold)
        st.image(visualized_output.get_image()[:, :, ::-1], caption='Inference time: {:.2f}s'.format(time.time() - start_time))
        st.write('Number of instances: {}'.format(len(predictions['instances'])))