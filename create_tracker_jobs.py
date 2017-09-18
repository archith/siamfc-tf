import os, sys, argparse
import pandas as pd
import cPickle
from glob import glob
import numpy as np
import glog
from multiprocessing import Pool
import shutil

videos_root = '/media/brain/archith/video_analysis/thumos14/subsampled_images_original_fps/'
detection_data_root = '/media/brain/archith/video_analysis/thumos14/object_detection_results/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/data/'
tracking_data_root = '/media/brain/archith/video_analysis/thumos14/object_tracking_results_unconsolidated/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/data/'

if not os.path.isdir(tracking_data_root):
  os.makedirs(tracking_data_root)

job_file = 'run_tracker_thumos14.sh'
with open(job_file, 'w') as f:
  f.write('#!/bin/bash\n')

tracker_job_pkls_root = '/run/shm/archith/video_analysis/thumos14/tracker_jobs'
if not os.path.isdir(tracker_job_pkls_root):
  os.makedirs(tracker_job_pkls_root)
else:
  shutil.rmtree(tracker_job_pkls_root)
  os.makedirs(tracker_job_pkls_root)

if not os.path.isdir(tracking_data_root):
  os.makedirs(tracking_data_root)



dataset = 'validation'
video_names = os.listdir(os.path.join(videos_root, dataset))
video_names.sort()
max_track_seq_len = 40
available_gpus = [0, 1, 2, 3]
num_gpus = len(available_gpus)
track_chunk_size = 200

def process_video(vid):
  glog.info(vid)
  vid_det_file = os.path.join(detection_data_root, '%s.csv'%vid)
  vid_det_df = pd.read_csv(vid_det_file, index_col=0)
  vid_img_root = os.path.join(videos_root, dataset, vid)
  image_paths = glob(os.path.join(vid_img_root, '*.png'))
  image_paths.sort()
  vid_num_frames = len(image_paths)
  det_frame_idxs = vid_det_df['image_idx'].tolist() # frames with detection bbox on them
  X = set(det_frame_idxs); det_frame_idxs = list(X)
  det_frame_idxs.sort()
  det_frame_idxs = det_frame_idxs[::2] # Subsampling to every 10 frame's detection to speed up

  result_df = pd.DataFrame(columns=vid_det_df.columns)
  result_df['box_source'] = []

  track_counter = 1

  video_job_cmds = []
  track_init_details_list = []

  for frame_idx in det_frame_idxs:
      num_frames_track_sequence = min(vid_num_frames-frame_idx, max_track_seq_len)
      track_seq_frame_idxs = range(frame_idx, frame_idx + num_frames_track_sequence)
      frame_name_list = [image_paths[i] for i in track_seq_frame_idxs]
      bbox_info_cols = vid_det_df.columns
      
      selection_crit = (vid_det_df['image_idx'] == frame_idx) & (vid_det_df['score'] > 0.5)
      frame_selected_bboxes = vid_det_df[selection_crit][bbox_info_cols]

      for row_idx, bbox_row in frame_selected_bboxes.iterrows():

          bbox_width = bbox_row['box_xmax'] - bbox_row['box_xmin']
          bbox_height = bbox_row['box_ymax'] - bbox_row['box_ymin']
          bbox_coords = np.array([bbox_row['box_xmin']*bbox_row['width'],
                          bbox_row['box_ymin']*bbox_row['height'],
                          bbox_width*bbox_row['width'],
                          bbox_height*bbox_row['height']])

          track_init_details = {}
          track_init_details['video_name'] = vid
          track_init_details['frame_name_list'] = frame_name_list
          track_init_details['bbox_coords'] = bbox_coords
          track_init_details['num_frames_track_sequence'] = num_frames_track_sequence
          track_init_details['track_seq_frame_idxs'] = track_seq_frame_idxs
          track_init_details['gpu_id'] = available_gpus[track_counter%num_gpus]
          track_init_details['track_counter'] = track_counter
          track_init_details['tracking_data_root'] = tracking_data_root
          track_init_details['detection_data_root'] = detection_data_root
          track_init_details['tracking_data_root'] = tracking_data_root
          track_init_details['frame_width'] = bbox_row['width']
          track_init_details['frame_height'] = bbox_row['height']
          track_init_details['score'] = bbox_row['score']
          track_init_details['class'] = bbox_row['class']

          track_init_details_list.append(track_init_details)
          track_counter += 1

  track_init_details_chunks = [track_init_details_list[i:i+track_chunk_size] 
                              for i in range(0, len(track_init_details_list), 
                              track_chunk_size)]

  for chunk_idx in range(0, len(track_init_details_chunks)):
    pkl_file_path = os.path.join(tracker_job_pkls_root, '%s_%06d.pkl'%(vid, chunk_idx))
    with open(pkl_file_path, 'wb') as f:
      cPickle.dump(track_init_details_chunks[chunk_idx], f, protocol=cPickle.HIGHEST_PROTOCOL)

    job_cmd = 'python ' + 'run_tracker_evaluation_thumos14.py ' + '--tracker-init-file=%s '%pkl_file_path
    video_job_cmds.append(job_cmd)

          
    
  return video_job_cmds


p = Pool(12)
results = p.map(process_video, [x for x in video_names])


all_cmds = []
for r in results:
  for job_cmd in r:
    all_cmds.append(job_cmd)

for idx, cmd in enumerate(all_cmds):
  cmd = cmd + '--gpu-id=%d'%available_gpus[idx%num_gpus]
  if (idx+1) % (2*num_gpus) == 0:
    cmd = cmd + '\nwait\n'
  else:
    cmd = cmd + ' &\n'

  all_cmds[idx] = cmd      
  

with open(job_file, 'a') as f:
  f.writelines(all_cmds)


glog.info('Finished!!')


