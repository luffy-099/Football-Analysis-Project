from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import numpy as np
import sys
import pandas as pd
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_positition_to_track(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #Interpolate missing ball positions
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox':x}}for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf = 0.1)
            detections += detections_batch 
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            'players': [], 
            'ball': [],
            'referee': []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            print(cls_names)

            #Convert to supervisio Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert goalkeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            #Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['ball'].append({})
            tracks['referee'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if track_id < 0:
                    continue #Skip Invalide ids

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox':bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks['referee'][frame_num][track_id] = {'bbox':bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox':bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width/2), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text), int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2)

        return frame

    def draw_triangle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, -1)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # Draw the border of the triangle
        
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw semi-transparent rectangle for team ball control
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)  
        alpha = 0.4  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        #Get the number of times each team had ball
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        total = team_1_num_frames + team_2_num_frames
        if total == 0:
            team_1 = 0
            team_2 = 0
        else:
            #Calculate the percentage of frames each team had the ball
            team_1 = team_1_num_frames/ total 
            team_2 = team_2_num_frames/ total

        cv2.putText(frame, f"Team 1: {team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2: {team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame


    def draw_annotations(self,video_frames,tracks,team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referee'][frame_num]

            #Draw players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))  # Default to red if no team color
                frame = self.draw_ellipse(frame, player['bbox'],color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 0, 255))  # Draw triangle for player with ball

            #Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'],(0,255,255), track_id)            

            #Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0,255,0))
            
            #Draw Possession
            frame = self.draw_team_ball_control(frame, frame_num,team_ball_control)

            # Draw the frame number
            output_video_frames.append(frame)
        
        return output_video_frames
    
    def interpolate_tracks(self, tracks, key='players'):
        """
        Linearly interpolate missing object positions in tracks[key].
        Works for 'players', 'ball', or 'referee'.
        """
        frames = tracks[key]
        track_ids = set()
        for frame_dict in frames:
            track_ids.update(frame_dict.keys())

        for track_id in track_ids:
            # Collect all frames where this track_id exists
            present = [(i, frame_dict[track_id]['bbox']) for i, frame_dict in enumerate(frames) if track_id in frame_dict]
            if len(present) < 2:
                continue  # Need at least two detections to interpolate

            for idx in range(len(present) - 1):
                start_frame, start_bbox = present[idx]
                end_frame, end_bbox = present[idx + 1]
                gap = end_frame - start_frame
                if gap > 1:
                    # Interpolate for missing frames
                    for j in range(1, gap):
                        interp_bbox = [
                            start_bbox[k] + (end_bbox[k] - start_bbox[k]) * j / gap
                            for k in range(4)
                        ]
                        frames[start_frame + j][track_id] = {'bbox': interp_bbox}
        return tracks