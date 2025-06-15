from sklearn.cluster import KMeans
import numpy as np
import cv2

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # 1: team 1, 2: team 2

    def get_clustering_model(self, image):
        # Reshape the image to a 2D array of pixels
        image_2d = image.reshape(-1, 3)

        # Use KMeans to find the most common colors - 2 clusters
        kmeans = KMeans(n_clusters=2, init ="k-means++")
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half = image[:int(image.shape[0] // 2), :]
        # Focus on central region to avoid background
        h, w, _ = top_half.shape
        shirt_region = top_half[h//4:h*3//4, w//4:w*3//4, :]
        pixels = shirt_region.reshape(-1, 3)
        mean_color = np.mean(pixels, axis=0)
        return mean_color

    def assign_team_color(self, video_frames, players_tracks, num_frames=20):
        player_colors = []
        for i in range(min(num_frames, len(players_tracks))):
            for _, player_detection in players_tracks[i].items():
                bbox = player_detection['bbox']
                player_color = self.get_player_color(video_frames[i], bbox)
                player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1  # Adjusting to 1-based index

        self.player_team_dict[player_id] = team_id

        return team_id