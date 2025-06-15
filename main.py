from utils import read_video, save_video, FeatureExtractor
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
import numpy as np

def main():
    # Read video
    video_frames = read_video("input_videos/trim_2.mp4")

    # Initialize tracker
    tracker = Tracker("models/best_yolovx.pt")
     # Initialize feature extractor
    feature_extractor = FeatureExtractor(device='cpu')

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
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
    
    # tracks = tracker.interpolate_tracks(tracks,key='players')
    # Interpolate player positions using embeddings
    tracks = tracker.interpolate_tracks_with_embeddings(
        tracks, video_frames, key='players', extract_embedding=feature_extractor
    )

    #Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    #Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks['players'])

    for frame_num in range(min(len(video_frames), len(tracks['players']))):
        player_tracks = tracks['players'][frame_num]
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
    for frame_num in range(min(len(video_frames), len(tracks['players']))):
        player_track = tracks['players'][frame_num]
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
    save_video(output_video_frames, "output_videos/trim2_deepsort.avi")



if __name__ == "__main__":
    main()