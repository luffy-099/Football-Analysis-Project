import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2

class HeatmapGenerator:
    def __init__(self, court_length=105, court_width=65, bin_size=1):
        self.court_length = court_length
        self.court_width = court_width
        self.bin_size = bin_size

    def generate_total_player_heatmap(self, tracks):
        heatmap = np.zeros((int(self.court_width/self.bin_size), int(self.court_length/self.bin_size)))

        for frame in tracks['players']:
            for player_id, player_data in frame.items():
                pos = player_data.get('position_transformed')
                if pos is None:
                    continue
                x, y = pos
                x_idx = int(x / self.bin_size)
                y_idx = int(y / self.bin_size)
                if 0 <= x_idx < heatmap.shape[1] and 0 <= y_idx < heatmap.shape[0]:
                    heatmap[y_idx, x_idx] += 1

        return heatmap

    def heatmap_to_image(self, heatmap, size=(200, 150)):
        fig, ax = plt.subplots(figsize=(4, 3))
        canvas = FigureCanvas(fig)
        ax.imshow(heatmap, cmap='hot', interpolation='nearest', origin='lower')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        img = cv2.resize(img, size)
        return img

