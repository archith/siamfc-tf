import os, sys, pandas as pd
import numpy as np
from glob import glob
import glog
from multiprocessing import Pool
import cv2

unconsolidated_data_root = '/media/ssd1/archith/video_analysis/thumos14/object_tracking_results_5.0_fps_unconsolidated/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/data/thumos14/'

consolidated_data_root = '/media/brain/archith/video_analysis/thumos14/object_tracking_results_5.0_fps_consolidated/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/data/thumos14/'
if not os.path.isdir(consolidated_data_root):
  os.makedirs(consolidated_data_root)

videos_root = '/media/ssd1/archith/video_analysis/thumos14/subsampled_images_5.0_fps/thumos14'

def draw_rectangle(img, bbox, color=[0, 255, 0], img_shape=(180, 320, 3)):
  # bbox is of the format np.array([box_ymin, box_xmin, box_ymax, box_xmax])
  img_h, img_w, _ = img_shape
  cv2.rectangle(img, (int(img_w*bbox[1]), int(img_h*bbox[0])), (int(img_w*bbox[3]), int(img_h*bbox[2])), color=color)
  return img

font = cv2.FONT_HERSHEY_SIMPLEX

def draw_string(img, char_string, bbox, color=[0, 255, 0], img_shape=(180, 320, 3)):
  # bbox is of the format np.array([box_ymin, box_xmin, box_ymax, box_xmax])
  img_h, img_w, _ = img_shape
  location = (int(bbox[1]*img_w) + 5, int(bbox[2]*img_h) + 5)
  cv2.putText(img, char_string, location, fontFace=font, fontScale=0.25, color=color, thickness=1)
  return img





video_names = os.listdir(videos_root)
video_names.sort()

video_names = video_names[0:1]

def measure_iou(bbox1, bbox2):
  # bboxes are of the format np.array([box_ymin, box_xmin, box_ymax, box_xmax])
  area_bbox1 = (bbox1[2] - bbox1[0])*(bbox1[3] - bbox1[1])
  area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

  intersection_w = max(min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]), 0)
  intersection_h = max(min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]), 0)

  return (intersection_w*intersection_h)/(area_bbox1 + area_bbox2 - intersection_w*intersection_h)

def process_video(vid):
  glog.info(vid)
  vid_track_files = glob(os.path.join(unconsolidated_data_root, '%s*.csv'%vid))
  vid_track_files.sort()

  vid_img_files = glob(os.path.join(videos_root, vid, 'img_*.jpg'))
  vid_img_files.sort()

  df_list = []
  for track_idx, file_path in enumerate(vid_track_files):
    file_df = pd.read_csv(file_path, index_col=False)
    file_df['track_number'] = [track_idx]*len(file_df)
    file_df = file_df[file_df.columns.tolist()[1:]] # Ignoring first header
    df_list.append(file_df)

  vid_df = pd.concat(df_list)
  
  # Filter out the tracks where there is poor overlap with detection output in intermediate frames
  track_idxes = vid_df['track_number'].tolist();
  tmp = set(track_idxes); track_idxes = list(tmp)
  track_idxes.sort()

  accepted_tracks = []
  for track_idx in track_idxes:
    track_df = vid_df[vid_df['track_number'] == track_idx]
    track_img_idxs = track_df['image_idx'].tolist()
    track_class = track_df['class'].tolist()[0]
    same_class = vid_df['class'] == track_class

    track_rows_with_dets = [] # list of track frames with a relevant detection output
    for img_idx in track_img_idxs:
      same_frame = vid_df['image_idx'] == img_idx
      det_source = vid_df['box_source'] == 'detection'

      criterion = same_class & same_frame & det_source
      
      track_rows_with_dets.append(vid_df[criterion])

    # Consolidate all the detection results which have common frames with the track
    track_det_df = pd.concat(track_rows_with_dets)
    track_det_image_idxes = list(set(track_det_df['image_idx'].tolist()))
    track_det_image_idxes.sort()

    # Find the number of frames over which detection of any class has been carried out
    num_track_frames_det = 0
    for img_idx in track_img_idxs:
      same_frame = vid_df['image_idx'] == img_idx
      det_source = vid_df['box_source'] == 'detection'
      criterion = same_frame & det_source
      if len(vid_df[criterion] ) > 0:
        num_track_frames_det += 1


    track_IOUs = []
    tracked_imgs = []

    for img_idx in track_det_image_idxes:
      img_det_df = track_det_df[track_det_df['image_idx'] == img_idx]

      img = cv2.imread(vid_img_files[img_idx])
      img_h, img_w, _ = img.shape

      track_img_row = track_df[track_df['image_idx'] == img_idx]
      track_bbox = np.squeeze(track_img_row[['box_ymin', 'box_xmin', 'box_ymax', 'box_xmax']].as_matrix()).astype(np.float32)

      iou_vals = []
      for _, row in img_det_df.iterrows():
        det_bbox = row[['box_ymin', 'box_xmin', 'box_ymax', 'box_xmax']].as_matrix().astype(np.float32)
        iou_vals.append(measure_iou(det_bbox, track_bbox))

        img = draw_rectangle(img, det_bbox, color=[255, 0, 0], img_shape=img.shape) # blue
        img = draw_rectangle(img, track_bbox, color=[0, 255, 0], img_shape=img.shape) # green

        iou_str = '{:.4f}'.format(measure_iou(det_bbox, track_bbox))
        img = draw_string(img, iou_str, det_bbox, [255, 0, 0], img_shape=img.shape)

      tracked_imgs.append(np.copy(img))


      max_IOU = max(iou_vals)
      track_IOUs.append(max_IOU)

    tracked_img = np.concatenate(tracked_imgs, axis=1)
    cv2.imshow(" track_idx : {:03d}".format(track_idx), tracked_img)
    cv2.waitKey(100)
    import pdb;pdb.set_trace()
    cv2.destroyAllWindows()

    if np.array(track_IOUs).sum() > 0.7*num_track_frames_det:
      # If the tracked frames overlap well in frames where detection has been carried out, 
      # then accept the track
      accepted_tracks.append(track_idx)


  accepted_track_dfs = []
  for track_idx in accepted_tracks:
    track_df = vid_df[vid_df['track_number'] == track_idx]
    accepted_track_dfs.append(track_df)
  
  vid_df = pd.concat(accepted_track_dfs)

  csv_filepath = os.path.join(consolidated_data_root, '%s.csv'%vid)
  vid_df.to_csv(csv_filepath, index=False)
  glog.info("CSV written to %s"%csv_filepath)


for x in video_names:
  process_video(x)

#p = Pool(24)
#p.map(process_video, [x for x in video_names])
#glog.info('Finished!!')