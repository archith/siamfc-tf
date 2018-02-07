from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox

import pandas as pd
from glob import glob
import numpy as np
from multiprocessing import Pool, current_process, Lock
import logging
import traceback
import glog
import argparse
import cPickle

parser = argparse.ArgumentParser(description='Run SIAMFC tracker on individual detection results for TH14')
parser.add_argument('--tracker-init-file', action="store", dest='tracker_init_file',
                    help="File containing tracking initialization information ")
parser.add_argument('--gpu-id', action="store", dest='gpu_id', default=0,
                    help="GPU ID ")

errorlog_root = 'errors'

args = parser.parse_args()



def process_track(track_init_details_list):
    
  

    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    # build TF graph once for all
    filename, image, templates_z, scores = siam.build_tracking_graph(final_score_sz, design, env) 
    
    try:
        frame_name_list_list = []
        pos_x_list = []
        pos_y_list = []
        target_w_list = []
        target_h_list = []

        detection_data_root = track_init_details_list[0]['detection_data_root']
        vid_det_file = os.path.join(detection_data_root, '%s.csv'%track_init_details_list[0]['video_name'])
        vid_det_df = pd.read_csv(vid_det_file, index_col=0)
        result_df_template = pd.DataFrame(columns=vid_det_df.columns)
        result_df_template['box_source'] = []
        result_df_cols = result_df_template.columns.tolist()

        for idx, track_init_details in enumerate(track_init_details_list):
            vid = track_init_details['video_name']
            frame_name_list = track_init_details['frame_name_list']
            bbox_coords = track_init_details['bbox_coords']
            num_frames_track_sequence = track_init_details['num_frames_track_sequence']
            track_seq_frame_idxs = track_init_details['track_seq_frame_idxs']
            track_idx = track_init_details['track_counter']
            tracking_data_root = track_init_details['tracking_data_root']
            detection_data_root = track_init_details['detection_data_root']
            frame_width = float(track_init_details['frame_width'])
            frame_height = float(track_init_details['frame_height'])
            bbox_score = track_init_details['score']
            bbox_class = track_init_details['class']
            result_csv_file = os.path.join(tracking_data_root, '%s_%06d.csv'%(vid, track_idx))
            
            glog.info(vid + ' track idx : %d'%(track_idx))
            if os.path.exists(result_csv_file):
                # already computed
                continue  
            


          

            gt = np.tile(bbox_coords, (num_frames_track_sequence, 1))
            pos_x, pos_y, target_w, target_h = region_to_bbox(gt[0])

            frame_name_list_list.append(frame_name_list)
            pos_x_list.append(pos_x)
            pos_y_list.append(pos_y)
            target_w_list.append(target_w)
            target_h_list.append(target_h)
        
        result_bboxes_list, speed_list = tracker(hp, run, design, frame_name_list_list, 
                                    pos_x_list, pos_y_list, target_w_list, target_h_list, 
                                    final_score_sz, filename, image, 
                                    templates_z, scores, 0, args.gpu_id)

        present_bboxes_idx = 0
        for idx, track_init_details in enumerate(track_init_details_list):
            vid = track_init_details['video_name']
            track_idx = track_init_details['track_counter']
            frame_width = float(track_init_details['frame_width'])
            frame_height = float(track_init_details['frame_height'])
            num_frames_track_sequence = track_init_details['num_frames_track_sequence']
            track_seq_frame_idxs = track_init_details['track_seq_frame_idxs']
            bbox_score = track_init_details['score']
            bbox_class = track_init_details['class']
            tracking_data_root = track_init_details['tracking_data_root']
            result_csv_file = os.path.join(tracking_data_root, '%s_%06d.csv'%(vid, track_idx))
            if os.path.exists(result_csv_file):
                # already computed
                glog.info(vid + ' track idx : %d'%(track_idx) \
                        + ' : Already computed')
                continue
            else:
                result_bboxes = result_bboxes_list[present_bboxes_idx]
                glog.info(vid + ' track idx : %d'%(track_idx) \
                        + ' : Speed: ' + "%.2f" % speed_list[present_bboxes_idx])
                present_bboxes_idx += 1                        
            

            result_df = pd.DataFrame(columns=result_df_cols)
            result_df['box_xmin'] =  result_bboxes[:, 0]/frame_width
            result_df['box_xmax'] =  (result_bboxes[:, 0] + result_bboxes[:, 2])/frame_width
            result_df['box_ymin'] =  result_bboxes[:, 1]/frame_height
            result_df['box_ymax'] =  (result_bboxes[:, 1] + result_bboxes[:, 3])/frame_height
            result_df['video_name'] = [vid]*num_frames_track_sequence
            result_df['image_idx'] = track_seq_frame_idxs
            result_df['score'] = [bbox_score]*num_frames_track_sequence
            result_df['class'] = [bbox_class]*num_frames_track_sequence
            result_df['width'] = [frame_width]*num_frames_track_sequence
            result_df['height'] = [frame_height]*num_frames_track_sequence

        

            if num_frames_track_sequence <= 1:
                result_df['box_source'] = ['detection']
            else:
                result_df['box_source'] = ['detection'] + \
                                        ['tracking']*(num_frames_track_sequence-1)

            result_df.to_csv(result_csv_file)
    except Exception as e:
        glog.error('Error occured while processing track : %s_%06d '%(vid, track_idx))
        glog.error(logging.error(traceback.format_exc()))
        errorlog_file = os.path.join(errorlog_root, '%s_%06d.log'%(vid, track_idx))
        with open(errorlog_file, 'w') as f:
            f.write('Error occured while processing track : %s_%06d '%(vid, track_idx))
            f.write(logging.error(traceback.format_exc()))


with open(args.tracker_init_file) as f:
    track_init_details_list = cPickle.load(f)

process_track(track_init_details_list)
