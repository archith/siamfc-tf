import os, sys, pandas as pd
import numpy as np
from glob import glob
import glog
from multiprocessing import Pool

unconsolidated_data_root = '/media/brain/archith/video_analysis/thumos14/object_tracking_results_unconsolidated/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/data/'

consolidated_data_root = '/media/brain/archith/video_analysis/thumos14/object_tracking_results_consolidated/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/data/'
if not os.path.isdir(consolidated_data_root):
  os.makedirs(consolidated_data_root)

videos_root = '/media/brain/archith/video_analysis/thumos14/subsampled_images_original_fps/'



dataset = 'validation'
video_names = os.listdir(os.path.join(videos_root, dataset))
video_names.sort()

def process_video(vid):
  glog.info(vid)
  vid_track_files = glob(os.path.join(unconsolidated_data_root, '%s*.csv'%vid))
  vid_track_files.sort()

  df_list = []
  for track_idx, file_path in enumerate(vid_track_files):
    file_df = pd.read_csv(file_path, index_col=False)
    file_df['track_number'] = [track_idx]*len(file_df)
    file_df = file_df[file_df.columns.tolist()[1:]] # Ignoring first header
    df_list.append(file_df)
  
  vid_df = pd.concat(df_list)
  csv_filepath = os.path.join(consolidated_data_root, '%s.csv'%vid)
  vid_df.to_csv(csv_filepath, index=False)
  glog.info("CSV written to %s"%csv_filepath)

#process_video(video_names[0])
p = Pool(24)
p.map(process_video, [x for x in video_names])
glog.info('Finished!!')