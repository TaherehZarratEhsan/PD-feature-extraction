from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score, balanced_accuracy_score, classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lightgbm as lgb
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks
import ast
from sklearn.utils import resample
import csv
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap
from math import pi
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd


sns.set(style="whitegrid", palette="coolwarm", font_scale=1.2)
MARGIN = 10 
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (255, 0, 0) 

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.results = {}
 
    
    
    def _init_hand_landmarker(self):
        model_path = r'/data/diag/Tahereh/new/src/Keypoint/hand_landmarker.task'
        with open(model_path, 'rb') as file:
            model_data = file.read()
        return model_data
      
    
    def _extract_features(self, distances):

        
        distances_ = np.array(distances)
       
        # Detect peaks (maxima) in the distance signal
        peaks, _ = find_peaks(distances_, height=np.mean(distances_), prominence=np.mean(distances_))
        
        # Detect troughs (minima) in the distance signal 
        troughs, _ = find_peaks(-distances_, height=-np.mean(distances_), prominence=np.mean(distances_))
        
        # Compute speed signal
        time_interval = 1 / self.fps
        speed_signal = np.diff(distances_) / time_interval # Speed = Δdistance / Δtime
            
        time_points_amp = np.arange(len(distances_)).reshape(-1, 1)
        model_amp = LinearRegression()
        model_amp.fit(time_points_amp, distances_)
        amp_slope = model_amp.coef_[0]
        
        time_points_speed = np.arange(len(speed_signal)).reshape(-1, 1)
        model_speed = LinearRegression()
        model_speed.fit(time_points_speed, speed_signal)
        speed_slope = model_speed.coef_[0]
        
        # Compute amplitudes
        amplitudes = []
        min_length = min(len(peaks), len(troughs))
        for i in range(min_length):
            if troughs[i] < peaks[i]:
                amplitude = distances_[peaks[i]] - distances_[troughs[i]]
            else:
                amplitude = distances_[troughs[i]] - distances_[peaks[i]]
            amplitudes.append(amplitude)
        
        # Compute median and max amplitude
        median_amplitude = np.median(amplitudes)
        max_amplitude = np.max(amplitudes)
        
        # Generate per-cycle speed maxima
        per_cycle_speed_maxima = []
        for i in range(len(amplitudes) - 1):
            start_idx = peaks[i]  # Start of the window
            end_idx = peaks[i + 1]  # End of the window
            window_speed = speed_signal[start_idx:end_idx]  # Slice the speed signal
            
            per_cycle_speed_maxima.append(np.max(window_speed))
        
        # Compute the median and max of per-cycle speed maxima
        median_speed = np.median(per_cycle_speed_maxima)
        max_speed = np.max(per_cycle_speed_maxima)




        # Compute tapping intervals (time between consecutive maxima)
        tapping_intervals = []
        for i in range(len(peaks) - 1):
            interval = (peaks[i + 1] - peaks[i]) * time_interval 
            tapping_intervals.append(interval)
        
        median_tapping_interval = np.median(tapping_intervals)
        mean_tapping_interval = np.mean(tapping_intervals)



        # Compute IQR/Median for amplitude
        amp_iqr = np.percentile(amplitudes, 75) - np.percentile(amplitudes, 25)
        amp_iqr_to_median = amp_iqr / median_amplitude 
        
        # Compute IQR/Median for speed
        speed_iqr = np.percentile(per_cycle_speed_maxima, 75) - np.percentile(per_cycle_speed_maxima, 25)
        speed_iqr_to_median = speed_iqr / median_speed 
        
        # Compute IQR/Median for tapping intervals
        tap_iqr = np.percentile(tapping_intervals, 75) - np.percentile(tapping_intervals, 25)
        tap_iqr_to_median = tap_iqr / median_tapping_interval 

        # Compute total number of interruptions
        threshold = 1.5 * median_tapping_interval
        num_interruptions = sum(interval > threshold for interval in tapping_intervals)
        

        
                
        self.feat_name= ['ids', 'video_path', 'label',
                        'median_amplitude', 'max_amplitude',
                        'median_speed', 'max_speed', 'median_tapping_interval',
                        'amp_slope','speed_slope',
                        'amp_iqr_to_median', 'speed_iqr_to_median', 'tap_iqr_to_median', 'num_interruptions']     
 
        features = [median_amplitude, max_amplitude,
                    median_speed, max_speed, median_tapping_interval, 
                    amp_slope, speed_slope, 
                    amp_iqr_to_median, speed_iqr_to_median,  tap_iqr_to_median,  num_interruptions]

  
        
        return features




    def _normalize_keypoints_distance(self, hand_landmarks):
        wrist = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
        index_mcp = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]

        wrist_to_index_mcp_length = np.sqrt((index_mcp.x - wrist.x) ** 2 +
                                            (index_mcp.y - wrist.y) ** 2 +
                                            (index_mcp.z - wrist.z) ** 2)

        if wrist_to_index_mcp_length < 1e-5:
            wrist_to_index_mcp_length = 1e-5

        normalized_keypoints = []
        for landmark in hand_landmarks:
            centered_x = landmark.x - wrist.x
            centered_y = landmark.y - wrist.y
            centered_z = landmark.z - wrist.z

            normalized_x = centered_x / wrist_to_index_mcp_length
            normalized_y = centered_y / wrist_to_index_mcp_length
            normalized_z = centered_z / wrist_to_index_mcp_length

            normalized_keypoints.append((normalized_x, normalized_y, normalized_z))


        return normalized_keypoints
    
 
    def trim_irrelevant_actions(self, sequence):
        peaks, _ = find_peaks(sequence, height=np.mean(sequence), prominence=np.mean(sequence))
        
        start = peaks[0]
        end = peaks[-1]
        trimmed_sequence = sequence[start:end+1]
        removed_start = sequence[:start]
        removed_end = sequence[end+1:]

  
        return trimmed_sequence, removed_start, removed_end, peaks


    def preprocess_and_display_video(self, video_path, label):
        """Process video and display frames with calculated distances or angles."""
        hand_to_track = 'Right' if '2R' in video_path else 'Left' if '2L' in video_path else None

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        sequences = []
        total_frames = 0
        detected_frames = 0
        self.palm_length = []
        model_data = self._init_hand_landmarker()

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            num_hands=2,
            running_mode=VisionRunningMode.VIDEO
        )
        hand_landmarker = HandLandmarker.create_from_options(options)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Filter hands based on the handedness
            if detection_result.handedness:
                filtered_landmarks = []
                for idx, handedness in enumerate(detection_result.handedness):
                    if handedness[0].category_name == hand_to_track:

                            #if self._validate_hand_with_pose(frame, detection_result.hand_landmarks[idx],  handedness[0].category_name):
    
                                filtered_landmarks.append(detection_result.hand_landmarks[idx])
                            #else:
                               #print(f"Pose validation failed for {hand_to_track} hand. Skipping frame.")
                               #self.wrong_hands.append(video_path)



                detection_result.hand_landmarks = filtered_landmarks

            if detection_result.hand_landmarks:
                detected_frames += 1
                for hand_landmarks in detection_result.hand_landmarks:
                    if self.config['distance']:
                        normalized_keypoints = self._normalize_keypoints_distance(hand_landmarks)
                        index_finger_tip = normalized_keypoints[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        thumb_tip = normalized_keypoints[mp.solutions.hands.HandLandmark.THUMB_TIP]
                        sequence_value = np.sqrt((thumb_tip[0] - index_finger_tip[0]) ** 2 +
                                                 (thumb_tip[1] - index_finger_tip[1]) ** 2 +
                                                 (thumb_tip[2] - index_finger_tip[2]) ** 2)
                    else:
                        sequence_value = self.angle_signal(hand_landmarks, width, height)

                    sequences.append(sequence_value)


        if detected_frames / total_frames >= 0.5 :
            
            if self.config['trimmed']==False:# 
                sequences = np.array(sequences)
                return np.array(sequences)
            elif self.config['trimmed']==True:
                trimmed_sequences, removed_start, removed_end, peaks = self.trim_irrelevant_actions(sequences)
                return np.array(trimmed_sequences)

        else:
            print(f"Skipping video {video_path} as keypoints were not detected in more than 50% of the frames.")
            return None

        cap.release()
        hand_landmarker.close()
        cv2.destroyAllWindows()
              
                      
    def run_savefeatures(self):
        ids = pd.read_csv(self.config['ids'], header=None)
        id2vid = pd.read_csv(self.config['id2vid'], header=None)
        patient_ids = ids.values[1:]
        video_labels = pd.read_csv(self.config['vid2score'])

        videos = []
        labels = []
        ids = []
        for patient in patient_ids:
            video_list = ast.literal_eval(id2vid[id2vid[0] == patient[0]].iloc[0, 1])
            for video in video_list:
                matching_rows = video_labels[video_labels['video_path'].str.contains(video, regex=False)]
                if not matching_rows.empty:
                    for _, row in matching_rows.iterrows():
                        videos.append(row['video_path'])
                        labels.append(row['score'])
                        ids.append(patient)
            #videos = [path.replace('//chansey.umcn.nl', '/data').replace('\\', '/') for path in videos]
            
        
        features = []
        modified_ids = []
        modified_videos = []
        modified_labels = []
        for vid, idi , label in tqdm(zip(videos, ids, labels), total=len(videos), desc="Processing videos"):
            print(vid)
            #if vid=='//chansey.umcn.nl/diag/Tahereh/Video\Visit 3\POM3VD7336728\On_2L_cropped_square.MP4':
            distance = self.preprocess_and_display_video(vid, label)
            feature = self._extract_features(distance)     
            features.append(feature)
            modified_ids.append(idi)
            modified_videos.append(vid)
            modified_labels.append(label)
        
        # Save to a CSV file
        csv_file_path = os.path.join(self.config['save_path'], 'video_features.csv')
        with open(csv_file_path, mode='w', newline='') as file:
              writer = csv.writer(file)
               
              writer.writerow(self.feat_name)
              
              
              
              
              # Write the rows
              for p_id, video_path, label, feature in zip(modified_ids, modified_videos, modified_labels, features):
                  writer.writerow([p_id] + [video_path] + [label] + feature)
          
      

if __name__ == "__main__":
    # Define configuration
    CONFIG = {
        'id2vid': r'/data/diag/Tahereh/new/src/datasets/dataset_preprocessing/id2vid.csv',
        'ids': r'/data/diag/Tahereh/new/src/datasets/dataset_preprocessing/patient_id_all.csv',
        'vid2score': r'/data/diag/Tahereh/new/src/datasets/dataset_preprocessing/ft_vid2score.csv',
        'save_path':  r'/data/diag/Tahereh/new/src1/my HC/id_based split/new_abs_speed',
        'distance': True,
        'trimmed':True,
    }

    # Instantiate the class and run the cross-validation
    trainer = ModelTrainer(CONFIG)
    trainer.run_savefeatures()

    
    
    
    
    
    
    
    
    