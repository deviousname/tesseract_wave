import pygame
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import rfft, irfft

class SignalProcessor:
    def __init__(self, file_path, recursion_depth=1):
        self.file_path = file_path
        self.recursion_depth = recursion_depth
        self.sample_rate, self.data = self.load_wav_file()
        self.fft_values = None
        self.original_data = None
        self.reconstructed_signal = None
        self.bands = []

    def load_wav_file(self):
        sample_rate, data = wavfile.read(self.file_path)
        if data.dtype == 'int16':
            nb_bits = 16
        elif data.dtype == 'int32':
            nb_bits = 32
        max_nb_bit = float(2 ** (nb_bits - 1))
        data = data / (max_nb_bit + 1.0)  # Normalize to -1.0 -- 1.0
        if data.ndim > 1:
            data = data[:, 0]  # Take only the first channel if stereo
        return sample_rate, data

    def perform_fft(self):
        self.fft_values = rfft(self.data)
        self.original_data = np.copy(self.data)

    def split_signal(self, fft_values, depth=1):
        energy = np.abs(fft_values)**2
        total_energy = np.sum(energy)
        cumulative_energy = np.cumsum(energy)
        half_energy_point = np.searchsorted(cumulative_energy, total_energy / 2)

        band_1_fft = np.zeros_like(fft_values)
        band_2_fft = np.zeros_like(fft_values)
        band_1_fft[:half_energy_point] = fft_values[:half_energy_point]
        band_2_fft[half_energy_point:] = fft_values[half_energy_point:]

        if depth < self.recursion_depth:
            self.split_signal(band_1_fft, depth + 1)
            self.split_signal(band_2_fft, depth + 1)
        else:
            band_1_signal = irfft(band_1_fft)
            band_2_signal = irfft(band_2_fft)
            self.bands.extend([band_1_signal, band_2_signal])

    def reconstruct_signal(self):
        self.reconstructed_signal = sum(self.bands)

    def process(self):
        self.perform_fft()
        self.split_signal(self.fft_values)
        self.reconstruct_signal()
        
def calculate_base_frequency(band):
    index_of_peak = np.argmax(np.abs(band))
    # Convert index to actual frequency using the sample rate and FFT size
    frequency = index_of_peak * processor.sample_rate / len(band)
    return frequency


# Colors
colors = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (192, 192, 192), # Silver
    (128, 0, 0),     # Maroon
    (128, 128, 0),   # Olive
    (0, 128, 0),     # Dark Green
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (0, 0, 128),     # Navy
    (255, 165, 0),   # Orange
    (255, 105, 180), # Hot Pink
    (75, 0, 130),    # Indigo
    (255, 192, 203), # Pink
    (64, 224, 208),  # Turquoise
    (255, 69, 0),    # Red Orange
    (47, 79, 79),    # Dark Slate Gray
    (0, 100, 0),     # Dark Green
    (72, 61, 139),   # Dark Slate Blue
    (143, 188, 143), # Dark Sea Green
    (210, 105, 30),  # Chocolate
    (188, 143, 143), # Rosy Brown
    (255, 228, 181), # Moccasin
    (250, 128, 114), # Salmon
    (244, 164, 96),  # Sandy Brown
    (32, 178, 170),  # Light Sea Green
    (95, 158, 160),  # Cadet Blue
    (100, 149, 237), # Cornflower Blue
    (70, 130, 180)   # Steel Blue
]

# Tesseract vertices in 4D space
vertices = [
    [1, 1, 1, 1], [1, 1, 1, -1], [1, 1, -1, 1], [1, 1, -1, -1],
    [1, -1, 1, 1], [1, -1, 1, -1], [1, -1, -1, 1], [1, -1, -1, -1],
    [-1, 1, 1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1], [-1, 1, -1, -1],
    [-1, -1, 1, 1], [-1, -1, 1, -1], [-1, -1, -1, 1], [-1, -1, -1, -1]
]

# Define edges between vertices
edges = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
    (4, 5), (4, 6), (5, 7), (6, 7), (0, 8), (1, 9), (2, 10), (3, 11),
    (4, 12), (5, 13), (6, 14), (7, 15), (8, 9), (8, 10), (8, 12), (9, 11),
    (9, 13), (10, 11), (10, 14), (11, 15), (12, 13), (12, 14), (13, 15), (14, 15)
]

def project4DTo2D(x, y, z, w, screen_width, screen_height, fov, viewer_distance):
    factor = fov / (viewer_distance + z)
    x = x * factor + screen_width / 2
    y = -y * factor + screen_height / 2
    return int(x), int(y)

def rotate4D(point, angle, axis1, axis2):
    rotation_matrix = np.identity(4)
    rotation_matrix[axis1, axis1] = rotation_matrix[axis2, axis2] = math.cos(angle)
    rotation_matrix[axis1, axis2] = -math.sin(angle)
    rotation_matrix[axis2, axis1] = math.sin(angle)
    return np.dot(point, rotation_matrix)

# Preprocess the audio to get the bands
processor = SignalProcessor('2.wav', recursion_depth=5) # depth 5 for 32 lines
processor.process()

# Get base frequencies from the bands
base_frequencies = [calculate_base_frequency(band) for band in processor.bands]

def draw_sine_wave(screen, start, end, center, color, time, width=2, max_amplitude = 32, base_frequency = None):
    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.sqrt(dx**2 + dy**2)
    num_points = max(int(distance / 2), 1)
    angle = math.atan2(dy, dx)

    points = []
    for i in range(num_points + 1):
        ratio = i / num_points
        x = start[0] + ratio * dx
        y = start[1] + ratio * dy

        # Calculate distance from the center for both the start and end point
        dist = math.sqrt((x - center[0])**2 + (y - center[1])**2)

        # Modify the frequency based on distance to the center
        frequency = base_frequency * (1 + (center[0] - dist) / center[0])

        # Calculate the phase of the wave so it starts at 0 at the first point and ends at π at the last point
        phase = math.pi * ratio

        # Offset y-coordinate by a sine wave that starts and ends at zero
        offset = max_amplitude * math.sin(frequency * i * 2 * math.pi / num_points + phase)
        offset_x = x + offset * math.cos(angle + math.pi / 2)
        offset_y = y + offset * math.sin(angle + math.pi / 2)
        
        if i > 0:
            pygame.draw.line(screen, color, (points[-1][0], points[-1][1]), (offset_x, offset_y), width)
        points.append((offset_x, offset_y))
        
# Function to change color dynamically
def get_dynamic_color(time, base_color, i):
    # Convert base_color to HSV
    color = pygame.Color(*base_color)
    hsva = color.hsva
    
    # Change hue over time
    new_hue = (hsva[0] + i * time * 10) % 360  # increment hue over time and loop around 360
    new_color = pygame.Color(0,0,0)
    new_color.hsva = (new_hue, hsva[1], hsva[2], hsva[3])
    
    return new_color

def main():
    # Initialize Pygame
    pygame.init()

    # Screen setup
    infoObject = pygame.display.Info()

    # Screen dimensions
    width, height = (1920,1080)#(int(infoObject.current_w/1.0), int(infoObject.current_h/1.0))
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    fov = 1000
    viewer_distance = 4
    rotationSpeed = 0.005

    # Initial rotation angles for each plane
    angleXY = angleXZ = angleXW = angleYZ = angleYW = angleZW = 0

    center_x = width // 2
    center_y = height // 2

    auto_rotate = False
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Resize the window
            elif event.type == pygame.VIDEORESIZE:
                print(f'screen resized {event.w}x{event.h}')
                width = event.w
                height = event.h
                center = (width//2, height//2)
            # Toggle fullscreen or maximize on key press
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:  # Fullscreen toggle with "F" key
                    pygame.display.toggle_fullscreen()

        screen.fill((0,0,0))
        keys = pygame.key.get_pressed()

        # Adjust rotation angles based on key presses
        if keys[pygame.K_UP]: angleXY += rotationSpeed
        if keys[pygame.K_DOWN]: angleXY -= rotationSpeed
        if keys[pygame.K_RIGHT]: angleXZ += rotationSpeed
        if keys[pygame.K_LEFT]: angleXZ -= rotationSpeed

        if keys[pygame.K_w]: angleXW += rotationSpeed
        if keys[pygame.K_s]: angleXW -= rotationSpeed
        if keys[pygame.K_a]: angleYZ += rotationSpeed
        if keys[pygame.K_d]: angleYZ -= rotationSpeed

        if keys[pygame.K_q]: angleYW += rotationSpeed
        if keys[pygame.K_e]: angleYW -= rotationSpeed

        if keys[pygame.K_r]: angleZW += rotationSpeed
        if keys[pygame.K_t]: angleZW -= rotationSpeed
        
        if keys[pygame.K_SPACE]: auto_rotate = not auto_rotate
        
        # Auto-update rotation angles
        if auto_rotate:
            angleXY += rotationSpeed # Slightly different speeds for more dynamic auto-rotation
            angleXZ += rotationSpeed * 0.5
            angleXW += rotationSpeed * 0.3
            angleYZ += rotationSpeed * 0.2
            angleYW += rotationSpeed * 0.1
            
        # Rotate and project vertices
        projected_vertices = []
        for v in vertices:
            # Apply rotations for each plane
            v = rotate4D(v, angleXY, 0, 1)
            v = rotate4D(v, angleXZ, 0, 2)
            v = rotate4D(v, angleXW, 0, 3)
            v = rotate4D(v, angleYZ, 1, 2)
            v = rotate4D(v, angleYW, 1, 3)
            v = rotate4D(v, angleZW, 2, 3)

            # Project to 2D
            projected = project4DTo2D(*v, width, height, fov, viewer_distance)
            projected_vertices.append(projected)

        # Draw edges
        time = pygame.time.get_ticks() / 1000  # Get time in seconds

        # Calculate center of the pygame window
        center = (center_x, center_y)

        # Draw the sine waves as the edges
        for i, edge in enumerate(edges):
            band_index = i % len(processor.bands)
            base_frequency = base_frequencies[band_index]
            dynamic_color = get_dynamic_color(time, colors[i % len(colors)], i)
            draw_sine_wave(screen, projected_vertices[edge[0]], projected_vertices[edge[1]], center, dynamic_color, time, base_frequency = base_frequency)
        
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    sys.exit()
if __name__ == "__main__":
    main()
