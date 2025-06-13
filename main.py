from utils import read_video, save_video, FeatureExtractor
from trackers import Tracker
import time
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
import numpy as np
from collections import Counter

def main():
    # Read video
    video_frames = read_video("input_videos/trim1.mp4")

    # Initialize tracker
    tracker = Tracker("models/best_tuned.pt")
     # Initialize feature extractor
    feature_extractor = FeatureExtractor(device='cpu')

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/track_stubs.pkl"
    )
    #Get object positions
    tracker.add_positition_to_track(tracks)

    # Camera Movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/camera_movement_stub.pkl"
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    

    # Interpolate player positions using embeddings
    tracks = tracker.interpolate_tracks_with_embeddings(
        tracks, video_frames, key='players', extract_embedding=feature_extractor
    )

    #Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    #Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            if player_id not in team_assigner.player_team_dict:
                team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            else:
                # Use the already assigned team
                team = team_assigner.player_team_dict[player_id]
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball to Player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control) == 0:
                team_ball_control.append(None)
            else:
                team_ball_control.append(team_ball_control[-1])
        
    team_ball_control = np.array(team_ball_control)


    #Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames,
        camera_movement_per_frame
    )

    # Save video
    save_video(output_video_frames, "output_videos/train1_possesion.avi")


    player_team_votes = {}

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            team = track.get('team')
            if team is not None:
                if player_id not in player_team_votes:
                    player_team_votes[player_id] = []
                player_team_votes[player_id].append(team)

    # Now, assign the most common team to each player in all frames
    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            if player_id in player_team_votes:
                most_common_team = Counter(player_team_votes[player_id]).most_common(1)[0][0]
                tracks['players'][frame_num][player_id]['team'] = most_common_team
                # Optionally, update team_color as well
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[most_common_team]


if __name__ == "__main__":
    main()