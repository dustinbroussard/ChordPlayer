import sys
import json
import time
import os
import hashlib
import threading
from typing import List, Dict, Tuple, Any, Optional

import pygame
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from mutagen import File as MutagenFile

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QSlider, QMessageBox, QProgressDialog
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QPalette

# --- Constants & Configuration ---
CHORDS_CACHE_DIR = "chord_cache"
CHORDS_CACHE_EXTENSION = ".chords.json"

# Audio processing parameters
AUDIO_SR = 22050  # Sample rate for internal processing and sounddevice stream
BUFFER_SIZE = 1024 # Audio buffer size for sounddevice callback (affect latency and FFT granularity)
FFT_SIZE = 2048    # NFFT for FFT, usually power of 2
HOP_LENGTH_CHORD_DETECTION = 1024 # Larger hop for chord detection (pre-analysis)
HOP_LENGTH_VISUALIZER = 512 # Smaller hop for real-time visualization

# Visualization parameters
PARTICLE_COUNT = 200
MAX_PARTICLE_SIZE = 8
MIN_PARTICLE_SIZE = 1
PARTICLE_VELOCITY_SCALE = 0.5
ORB_BASE_RADIUS = 80
ORB_MAX_RADIUS_ADD = 100
ORB_GLOW_RADIUS_MULT = 1.5
CHORD_TEXT_FADE_DISTANCE = 300 # Pixels from center for chord text to start fading

# --- Colors for Chord Visualization (RGB Tuples) ---
CHORD_COLORS = {
    # Major Chords (Warm, bright)
    "C": (255, 200, 0),    # Golden Yellow
    "C#": (255, 170, 0),   # Orange-Yellow
    "D": (150, 255, 0),    # Lime Green
    "D#": (80, 255, 0),    # Brighter Green
    "E": (255, 0, 150),    # Rose Pink
    "F": (0, 255, 200),    # Aqua
    "F#": (0, 255, 150),   # Greenish Aqua
    "G": (255, 100, 0),    # Bright Orange
    "G#": (255, 50, 0),    # Deep Orange
    "A": (0, 150, 255),    # Sky Blue
    "A#": (0, 100, 255),   # Royal Blue
    "B": (200, 0, 255),    # Purple Magenta

    # Minor Chords (Cool, sometimes melancholic)
    "Cm": (100, 100, 255),  # Medium Blue
    "C#m": (80, 80, 200),   # Deeper Blue
    "Dm": (50, 200, 50),    # Forest Green
    "D#m": (0, 150, 0),     # Dark Green
    "Em": (200, 50, 100),   # Muted Red-Pink
    "Fm": (0, 100, 150),    # Dark Cyan
    "F#m": (0, 80, 120),    # Even darker Cyan
    "Gm": (150, 50, 0),     # Burnt Orange
    "G#m": (100, 0, 0),     # Deep Red
    "Am": (0, 50, 150),     # Indigo
    "A#m": (0, 30, 100),    # Very Dark Blue
    "Bm": (150, 0, 200),    # Darker Purple

    # 7th Chords (More complex/vibrant)
    "C7": (255, 120, 0),    # Warmer Orange
    "C#7": (255, 90, 0),
    "D7": (100, 255, 50),
    "D#7": (50, 255, 0),
    "E7": (255, 0, 100),
    "F7": (0, 255, 100),
    "F#7": (0, 200, 80),
    "G7": (255, 80, 0),
    "G#7": (255, 30, 0),
    "A7": (0, 100, 200),
    "A#7": (0, 80, 180),
    "B7": (180, 0, 255),

    # Major 7th Chords
    "Cmaj7": (255, 220, 50), # Lighter Gold
    "Dmaj7": (180, 255, 50),
    "Emaj7": (255, 50, 180),
    "Fmaj7": (50, 255, 220),
    "Gmaj7": (255, 150, 50),
    "Amaj7": (50, 200, 255),
    "Bmaj7": (220, 50, 255),

    # Minor 7th Chords
    "Cm7": (120, 120, 255), # Slightly brighter minor blue
    "Dm7": (80, 220, 80),
    "Em7": (220, 80, 120),
    "Fm7": (50, 120, 180),
    "Gm7": (180, 80, 50),
    "Am7": (50, 80, 180),
    "Bm7": (180, 50, 220),

    # Diminished, Augmented, Suspended (Placeholder or fill as needed)
    "dim": (120, 80, 150), # Muted purple
    "aug": (255, 100, 255), # Pinkish purple
    "sus2": (100, 200, 200), # Light Teal
    "sus4": (200, 150, 100), # Orangish brown

    # Other / Fallback
    "N/A": (80, 80, 80), # Neutral Grey, for no clear chord
    "Unknown": (50, 50, 50), # Darker Grey, general fallback
}

def lerp_color(color1: Tuple[int, int, int], color2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Linear interpolation between two RGB colors."""
    t = max(0.0, min(1.0, t)) # Clamp t between 0 and 1
    r = int(color1[0] + (color2[0] - color1[0]) * t)
    g = int(color1[1] + (color2[1] - color1[1]) * t)
    b = int(color1[2] + (color2[2] - color1[2]) * t)
    return (r, g, b)

# --- Chord Detector Class (Async Pre-Analysis) ---
class ChordDetector(QThread):
    chords_detected = pyqtSignal(list, float) # Emits (chords_list, duration_seconds)
    detection_error = pyqtSignal(str)
    detection_progress = pyqtSignal(int) # Emits percentage of progress

    def __init__(self):
        super().__init__()
        self.audio_file: Optional[str] = None
        self.chord_templates = self._create_chord_templates()
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        os.makedirs(CHORDS_CACHE_DIR, exist_ok=True)

    def _create_chord_templates(self) -> Dict[str, np.ndarray]:
        """Create chord templates for major, minor, and 7th chords."""
        templates: Dict[str, np.ndarray] = {}
        # Intervals: (root, interval1, interval2, ...)
        chord_definitions: Dict[str, List[int]] = {
            "": [0, 4, 7],      # Major triad
            "m": [0, 3, 7],     # Minor triad
            "7": [0, 4, 7, 10], # Dominant 7th
            "maj7": [0, 4, 7, 11], # Major 7th
            "m7": [0, 3, 7, 10], # Minor 7th
            "dim": [0, 3, 6],   # Diminished triad
            "aug": [0, 4, 8],   # Augmented triad
            "sus2": [0, 2, 7],  # Suspended 2nd
            "sus4": [0, 5, 7],  # Suspended 4th
        }

        for root_idx, root_name in enumerate(self.note_names):
            for suffix, intervals in chord_definitions.items():
                template = np.zeros(12)
                for interval in intervals:
                    template[(root_idx + interval) % 12] = 1
                templates[f"{root_name}{suffix}"] = template
        return templates

    def set_audio_file(self, audio_file_path: str):
        self.audio_file = audio_file_path

    def run(self):
        if not self.audio_file or not os.path.exists(self.audio_file):
            self.detection_error.emit("No audio file specified for chord detection.")
            return

        file_hash = self._get_file_hash(self.audio_file)
        if file_hash is None: # Hashing failed, cannot use cache
            print("File hashing failed, skipping cache.")
            cache_file = None
        else:
            cache_file = os.path.join(CHORDS_CACHE_DIR, file_hash + CHORDS_CACHE_EXTENSION)

        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                self.chords_detected.emit(cached_data['chords'], cached_data['duration'])
                print(f"Loaded chords from cache: {cache_file}")
                return
            except Exception as e:
                print(f"Error loading from cache '{cache_file}': {e}. Re-detecting chords.")
                # If cache fails, proceed to re-detect

        try:
            self.detection_progress.emit(5)
            y, sr = librosa.load(self.audio_file, sr=AUDIO_SR, mono=True)
            
            audio_info = MutagenFile(self.audio_file)
            duration_seconds = audio_info.info.length if audio_info and audio_info.info else len(y) / sr

            self.detection_progress.emit(10)

            # Use chroma_cens which is more robust to timbre changes
            chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP_LENGTH_CHORD_DETECTION)
            
            self.detection_progress.emit(50)

            times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP_LENGTH_CHORD_DETECTION)

            raw_chords: List[Tuple[float, str]] = []
            for i in range(chroma.shape[1]):
                frame_chroma = chroma[:, i]
                frame_chroma = frame_chroma / (np.sum(frame_chroma) + 1e-8) # Normalize

                best_chord = self._match_chord(frame_chroma)
                raw_chords.append((times[i], best_chord))
                if i % 100 == 0: # Update progress more frequently for long songs
                    self.detection_progress.emit(50 + int(i / chroma.shape[1] * 40))

            processed_chords = self._smooth_chords(raw_chords, min_duration=0.6) # Slightly longer min duration for better stability

            self.detection_progress.emit(95)

            if cache_file: # Only save to cache if hashing was successful
                cached_data = {
                    'file_path': self.audio_file,
                    'duration': duration_seconds,
                    'chords': processed_chords
                }
                with open(cache_file, 'w') as f:
                    json.dump(cached_data, f, indent=2)
                print(f"Chords saved to cache: {cache_file}")

            self.chords_detected.emit(processed_chords, duration_seconds)
            self.detection_progress.emit(100)
        except Exception as e:
            self.detection_error.emit(f"Error during chord detection: {e}")
            print(f"Error during chord detection for {self.audio_file}: {e}")

    def _match_chord(self, chroma: np.ndarray, threshold: float = 0.25) -> str: # Adjusted threshold for CENS
        """Match chroma vector to closest chord template using cosine similarity."""
        best_match = "N/A"
        best_score = threshold

        for chord_name, template in self.chord_templates.items():
            norm_template = template / (np.sum(template) + 1e-8)
            # Add a small epsilon to avoid division by zero for silent frames/empty vectors
            denom = np.linalg.norm(chroma) * np.linalg.norm(norm_template) + 1e-8
            score = np.dot(chroma, norm_template) / denom

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_match = chord_name
        return best_match

    def _smooth_chords(self, raw_chords: List[Tuple[float, str]], min_duration: float = 0.5) -> List[Dict[str, Any]]:
        """
        Consolidates rapidly changing chords into more stable, longer-duration chords.
        'min_duration' is the minimum time a chord must be present to be considered significant.
        """
        if not raw_chords:
            return []

        smoothed_chords: List[Dict[str, Any]] = []
        current_chord_name = raw_chords[0][1]
        current_chord_start_time = raw_chords[0][0]

        for i in range(1, len(raw_chords)):
            time_stamp, chord_name = raw_chords[i]
            if chord_name != current_chord_name:
                if (time_stamp - current_chord_start_time) >= min_duration:
                    smoothed_chords.append({
                        "timestamp": int(current_chord_start_time * 1000),
                        "chord": current_chord_name
                    })
                # Start tracking the new chord
                current_chord_name = chord_name
                current_chord_start_time = time_stamp

        # Add the very last chord segment if it's significant or if there's only one chord
        if (raw_chords[-1][0] - current_chord_start_time) >= min_duration or not smoothed_chords:
            smoothed_chords.append({
                "timestamp": int(current_chord_start_time * 1000),
                "chord": current_chord_name
            })
        else: # If the last segment was too short, extend the previous one if it exists
            if smoothed_chords:
                smoothed_chords[-1]['timestamp_end'] = int(raw_chords[-1][0] * 1000)

        # Ensure the first chord is at timestamp 0 (crucial for visualizer)
        if smoothed_chords and smoothed_chords[0]['timestamp'] > 0:
            smoothed_chords.insert(0, {"timestamp": 0, "chord": smoothed_chords[0]['chord']})
        elif not smoothed_chords:
            smoothed_chords.append({"timestamp": 0, "chord": "N/A"}) # Default if no chords detected

        return smoothed_chords

    def _get_file_hash(self, file_path: str) -> Optional[str]:
        """Generates a SHA256 hash of the file content for caching."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"Error hashing file {file_path}: {e}. Skipping cache.")
            return None


# --- Live Audio Processor for Playback & Visualization Data ---
class LiveAudioProcessor(QThread):
    volume_data = pyqtSignal(float) # Emits RMS volume (0.0 to 1.0)
    freq_band_data = pyqtSignal(dict) # Emits dict with bass, mids, highs energy
    position_updated = pyqtSignal(int) # Emits current position in ms
    playback_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._audio_data: Optional[np.ndarray] = None
        self._current_song_path: Optional[str] = None
        self._stream: Optional[sd.OutputStream] = None
        self._play_pos_frame: int = 0
        self._is_playing: bool = False
        self._is_paused: bool = False
        self._lock = threading.Lock() # For thread-safe access to audio data

    def load_song(self, path: str) -> bool:
        with self._lock:
            self.stop() # Ensure previous stream is stopped
            if not os.path.exists(path):
                print(f"Error: Song file not found at {path}")
                return False
            
            try:
                # Use soundfile to load the entire audio (will be processed in chunks by callback)
                data, sr = sf.read(path, dtype='float32')
                # Ensure mono and correct sample rate
                if data.ndim > 1:
                    data = data.mean(axis=1) # Convert to mono
                if sr != AUDIO_SR:
                    # Librosa resample is robust. Use it directly if needed.
                    data = librosa.resample(data, orig_sr=sr, target_sr=AUDIO_SR)
                    sr = AUDIO_SR
                
                self._audio_data = data
                self._current_song_path = path
                self._play_pos_frame = 0
                self._is_playing = False
                self._is_paused = False
                print(f"Audio loaded for stream: {path}")
                return True
            except Exception as e:
                print(f"Error loading audio with soundfile: {e}")
                self._audio_data = None
                return False

    def play(self):
        with self._lock:
            if self._audio_data is None:
                print("No audio data to play.")
                return

            if not self._is_playing and not self._is_paused:
                # Start fresh
                self._play_pos_frame = 0
                self._stream = sd.OutputStream(
                    samplerate=AUDIO_SR,
                    channels=1, # Our audio_data is mono
                    blocksize=BUFFER_SIZE,
                    callback=self._audio_callback,
                    finished_callback=self._stream_finished
                )
                self._stream.start()
                self._is_playing = True
                print("Stream started (from beginning).")
            elif self._is_paused:
                if self._stream and not self._stream.active:
                    self._stream.start() # Resume stream
                self._is_playing = True
                self._is_paused = False
                print("Stream resumed.")

    def pause(self):
        with self._lock:
            if self._is_playing and not self._is_paused:
                if self._stream and self._stream.active:
                    self._stream.stop()
                self._is_playing = False
                self._is_paused = True
                print("Stream paused.")

    def stop(self):
        with self._lock:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._is_playing = False
            self._is_paused = False
            self._play_pos_frame = 0
            self.position_updated.emit(0)
            self.volume_data.emit(0.0) # Reset visualizer
            self.freq_band_data.emit({"bass": 0.0, "mids": 0.0, "highs": 0.0})
            print("Stream stopped and reset.")
    
    def set_position(self, ms: int):
        with self._lock:
            if self._audio_data is None:
                return
            frame = int(ms / 1000.0 * AUDIO_SR)
            self._play_pos_frame = min(frame, len(self._audio_data))

            was_playing = self._is_playing
            self.stop() # Stop and reset stream for reliable seek
            
            if was_playing or self._is_paused:
                self.play() # Restart if it was playing/paused

            self.position_updated.emit(self._play_pos_frame * 1000 // AUDIO_SR)


    def _audio_callback(self, outdata: np.ndarray, frames: int, time_status, status):
        if status:
            print(status)
        
        with self._lock:
            if self._audio_data is None or not self._is_playing:
                outdata.fill(0)
                return

            # Get current chunk of audio data
            chunk_start = self._play_pos_frame
            chunk_end = chunk_start + frames
            
            # Pad with zeros if we're near the end of the song
            if chunk_end > len(self._audio_data):
                samples_to_read = len(self._audio_data) - chunk_start
                outdata[:samples_to_read, 0] = self._audio_data[chunk_start:len(self._audio_data)]
                outdata[samples_to_read:, 0] = 0.0 # Fill remaining with zeros
                self._play_pos_frame = len(self._audio_data) # Mark as end
                self.playback_finished.emit() # Signal playback end
                
            else:
                outdata[:, 0] = self._audio_data[chunk_start:chunk_end]
                self._play_pos_frame += frames

            # --- Real-time Audio Analysis for Visualizer ---
            current_audio_chunk = outdata[:, 0] # Operate on the mono channel

            # 1. RMS Volume
            rms_vol = np.sqrt(np.mean(current_audio_chunk**2))
            self.volume_data.emit(float(rms_vol))

            # 2. FFT for Frequency Bands
            if len(current_audio_chunk) >= FFT_SIZE: # Ensure enough data for FFT
                # Apply Hann window to reduce spectral leakage
                windowed_chunk = current_audio_chunk[:FFT_SIZE] * np.hanning(FFT_SIZE)
                
                # Perform FFT
                fft_result = np.fft.fft(windowed_chunk)
                
                # Get magnitudes of positive frequencies
                magnitudes = np.abs(fft_result[:FFT_SIZE // 2])
                
                # Frequency bins (Hz)
                freqs = np.fft.fftfreq(FFT_SIZE, d=1/AUDIO_SR)[:FFT_SIZE // 2]

                # Define frequency bands (adjust as needed)
                bass_idx = np.where((freqs >= 20) & (freqs <= 250))[0]
                mids_idx = np.where((freqs > 250) & (freqs <= 4000))[0]
                highs_idx = np.where((freqs > 4000) & (freqs <= 20000))[0] # Up to human hearing limit

                bass_energy = np.sum(magnitudes[bass_idx])
                mids_energy = np.sum(magnitudes[mids_idx])
                highs_energy = np.sum(magnitudes[highs_idx])
                
                # Normalize energies (simple max normalization, can be improved with log scale)
                total_energy = bass_energy + mids_energy + highs_energy + 1e-8
                freq_data = {
                    "bass": bass_energy / total_energy,
                    "mids": mids_energy / total_energy,
                    "highs": highs_energy / total_energy,
                }
                self.freq_band_data.emit(freq_data)
            else:
                self.freq_band_data.emit({"bass": 0.0, "mids": 0.0, "highs": 0.0}) # No data for FFT

            # 3. Position Update
            current_ms = self._play_pos_frame * 1000 // AUDIO_SR
            self.position_updated.emit(current_ms)

    def _stream_finished(self):
        with self._lock:
            self._is_playing = False
            self._is_paused = False
            self._play_pos_frame = 0
            self.playback_finished.emit()
            print("Sounddevice stream finished.")
    
    def get_duration_ms(self) -> int:
        return int((len(self._audio_data) / AUDIO_SR) * 1000) if self._audio_data is not None else 0

    def run(self):
        # The QThread needs a run method, but sounddevice manages its own playback loop.
        # We just need to keep this thread alive while it's in use by the main app.
        # This can be achieved by not exiting until explicitly told to.
        # The stream's callbacks run on a separate native thread managed by sounddevice.
        self.exec() # Start PyQt's event loop for this thread (useful if we add QTimers here later)


# --- Pygame Visualizer Window ---
class VisualizerWindow(QWidget):
    # Signals for communication from Visualizer to Main App
    window_closed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Code Maniac Visualizer")
        self.setGeometry(100, 100, 1024, 768)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint) # Start without frame
        
        self.pygame_widget = PygameWidget(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.pygame_widget)
        self.setLayout(layout)
        
        # Initial states
        self.is_fullscreen = False
        self.volume = 0.0
        self.freq_data = {"bass": 0.0, "mids": 0.0, "highs": 0.0}
        self.current_chord_info: Dict[str, Any] = {"chord": "N/A", "color": CHORD_COLORS["N/A"]}
        self.chord_timeline: List[Dict[str, Any]] = [] # Full list of detected chords
        self.playback_position_ms = 0
        self.song_duration_ms = 0

        # Current and Target colors for smooth transition
        self.current_orb_color: Tuple[int, int, int] = CHORD_COLORS["N/A"]
        self.target_orb_color: Tuple[int, int, int] = CHORD_COLORS["N/A"]
        self.color_transition_start_time = time.time()
        self.color_transition_duration = 0.5 # seconds

        # Pygame loop timer for consistent updates (e.g., 60 FPS)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.pygame_widget.update_visuals)
        self.update_timer.start(16) # ~60 FPS

        # Set up keyboard shortcuts
        self.grabKeyboard() # Grab all keyboard events for global shortcuts

    def keyPressEvent(self, event: Any): # QKeyEvent type, using Any for simpler import
        if event.key() == Qt.Key.Key_F11 or event.key() == Qt.Key.Key_F:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Escape:
            if self.is_fullscreen:
                self.toggle_fullscreen()
            else:
                self.close() # Close window if not fullscreen
        super().keyPressEvent(event)

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.showNormal()
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            self.show() # Reapply flags
            self.is_fullscreen = False
            print("Exiting fullscreen.")
        else:
            self.showFullScreen()
            self.is_fullscreen = True
            print("Entering fullscreen.")

    def set_realtime_data(self, volume: float, freq_data: dict, current_pos_ms: int):
        self.volume = volume
        self.freq_data = freq_data
        self.playback_position_ms = current_pos_ms
        self._update_current_chord_and_color() # Update chord and color based on position
        
        # Pass all necessary data to the Pygame widget
        self.pygame_widget.set_visualizer_data(
            self.volume, self.freq_data, self.current_orb_color, 
            self.current_chord_info['chord'], self.chord_timeline, 
            self.playback_position_ms, self.song_duration_ms
        )

    def set_chord_timeline(self, timeline: List[Dict[str, Any]], duration_ms: int):
        self.chord_timeline = timeline
        self.song_duration_ms = duration_ms
        self._update_current_chord_and_color()

    def _update_current_chord_and_color(self):
        # Find the correct chord for the current timestamp
        new_chord_name = "N/A"

        if self.chord_timeline:
            for i in range(len(self.chord_timeline) - 1, -1, -1):
                if self.chord_timeline[i]['timestamp'] <= self.playback_position_ms:
                    new_chord_name = self.chord_timeline[i]['chord']
                    break
            
            # Special case for 0ms if it's the only chord or first chord isn't at 0
            if self.playback_position_ms == 0 and self.chord_timeline and self.chord_timeline[0]['timestamp'] > 0:
                 new_chord_name = self.chord_timeline[0]['chord']

        current_color_obj = CHORD_COLORS.get(new_chord_name, CHORD_COLORS["Unknown"])

        # Update target color if chord name changed
        if new_chord_name != self.current_chord_info['chord']:
            self.target_orb_color = current_color_obj
            self.current_chord_info['chord'] = new_chord_name
            self.color_transition_start_time = time.time() # Reset transition timer
            
        # Smooth color transition logic
        elapsed = time.time() - self.color_transition_start_time
        transition_progress = elapsed / self.color_transition_duration
        
        # Interpolate between the *previous actual* color and the *target* color
        # This requires tracking the actual color from the *start* of the transition
        # For simplicity in this example, we'll just lerp from the last frame's current_orb_color
        # if a new target is set. A more robust solution would store the color when transition began.
        if transition_progress < 1.0:
            self.current_orb_color = lerp_color(self.current_orb_color, self.target_orb_color, transition_progress)
        else:
            self.current_orb_color = self.target_orb_color # Ensure it snaps to target at end

        # Also pass the interpolated color to chord info for internal use / drawing
        self.current_chord_info['color'] = self.current_orb_color
        
    def closeEvent(self, event: Any): # QCloseEvent type, using Any
        self.update_timer.stop()
        self.window_closed.emit() # Notify main app
        super().closeEvent(event)


class PygameWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen) # Important for direct Pygame rendering

        self.screen: Optional[pygame.Surface] = None
        self.size = (self.width(), self.height())
        
        pygame.font.init()
        self.orb_font = pygame.font.Font(None, 48) # Default font, size dynamic
        self.chord_font = pygame.font.Font(None, 36) # For scrolling chords

        # Visualizer data (updated by VisualizerWindow)
        self.volume = 0.0
        self.freq_data = {"bass": 0.0, "mids": 0.0, "highs": 0.0}
        self.orb_color: Tuple[int, int, int] = CHORD_COLORS["N/A"]
        self.current_chord_name: str = "N/A"
        self.chord_timeline: List[Dict[str, Any]] = []
        self.playback_position_ms: int = 0
        self.song_duration_ms: int = 0

        # Particles (for the "interesting shit")
        self.particles: List[Dict[str, Any]] = []
        self._init_particles()
    
    def _init_particles(self):
        for _ in range(PARTICLE_COUNT):
            self.particles.append(self._create_particle())

    def _create_particle(self) -> Dict[str, Any]:
        cx, cy = self.width() // 2, self.height() // 2
        angle = np.random.uniform(0, 2 * np.pi)
        radius_offset = np.random.uniform(ORB_BASE_RADIUS, ORB_BASE_RADIUS + ORB_MAX_RADIUS_ADD * 0.5)
        return {
            "x": cx + np.cos(angle) * radius_offset,
            "y": cy + np.sin(angle) * radius_offset,
            "vx": np.random.uniform(-1, 1),
            "vy": np.random.uniform(-1, 1),
            "size": np.random.uniform(MIN_PARTICLE_SIZE, MAX_PARTICLE_SIZE),
            "alpha": 255, # Full opacity initially
            "life": np.random.uniform(100, 300) # Lifespan in updates
        }

    def set_visualizer_data(self, volume: float, freq_data: dict, orb_color: Tuple[int, int, int], 
                            current_chord_name: str, chord_timeline: List[Dict[str, Any]], 
                            playback_pos_ms: int, song_duration_ms: int):
        self.volume = volume
        self.freq_data = freq_data
        self.orb_color = orb_color
        self.current_chord_name = current_chord_name
        self.chord_timeline = chord_timeline
        self.playback_position_ms = playback_pos_ms
        self.song_duration_ms = song_duration_ms

    def sizeHint(self) -> QSize:
        return QSize(1024, 768)

    def minimumSizeHint(self) -> QSize:
        return QSize(640, 480)

    def paintEvent(self, event: Any): # QPaintEvent
        if self.screen is None:
            self.init_pygame_surface()
        self.update_visuals()

    def resizeEvent(self, event: Any): # QResizeEvent
        self.size = (self.width(), self.height())
        if self.screen:
            self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE | pygame.NOFRAME)
            # Re-initialize particles to new center
            self.particles = []
            self._init_particles()
        super().resizeEvent(event)

    def init_pygame_surface(self):
        # Set the environment variable for Wayland compatibility (if applicable)
        os.environ['SDL_VIDEODRIVER'] = 'wayland' # Or 'x11' if issues occur
        pygame.init()
        # Initialize pygame display for embedding in Qt
        # We use NOFRAME because Qt provides the window frame
        # If running standalone, you might use pygame.FULLSCREEN instead of pygame.NOFRAME and RESIZABLE
        self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE | pygame.NOFRAME)
        pygame.display.set_caption("Code Maniac Visualizer") # Will not show with NOFRAME
        pygame.event.set_grab(True) # Grab mouse for better fullscreen experience
        pygame.mouse.set_visible(False) # Hide mouse cursor

    def update_visuals(self):
        if self.screen is None:
            self.init_pygame_surface()
            return
        
        # --- Drawing ---
        self.screen.fill((0, 0, 0)) # Solid black background

        # Calculate dynamic orb properties
        volume_scaled = min(1.0, self.volume * 20.0) # Scale volume for visual effect
        dynamic_radius = ORB_BASE_RADIUS + ORB_MAX_RADIUS_ADD * volume_scaled
        glow_radius = int(dynamic_radius * ORB_GLOW_RADIUS_MULT)
        
        # Orb center
        cx, cy = self.width() // 2, self.height() // 2 - 50 # Slightly above center for chord text

        # Draw glowing orb (multiple concentric circles with fading alpha)
        for i in range(5, 0, -1):
            alpha = int(255 * (i / 5.0) * (volume_scaled + 0.1)) # More alpha with more volume
            color_with_alpha = (self.orb_color[0], self.orb_color[1], self.orb_color[2], alpha)
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, color_with_alpha, (glow_radius, glow_radius), int(glow_radius * (i/5.0)))
            self.screen.blit(glow_surface, (cx - glow_radius, cy - glow_radius))

        # Draw main orb
        pygame.draw.circle(self.screen, self.orb_color, (cx, cy), int(dynamic_radius))

        # --- Particle system (Frequency-driven) ---
        new_particles: List[Dict[str, Any]] = []
        highs_scaled = self.freq_data.get("highs", 0.0) * 100 # Scale for effect

        # Spawn new particles based on highs
        for _ in range(int(highs_scaled / 10 + 1)): # More particles for higher energy
            if len(self.particles) < PARTICLE_COUNT * 2: # Max particles
                self.particles.append(self._create_particle())

        for p in self.particles:
            # Move particles based on velocity and some randomness/bass impact
            p["x"] += p["vx"] * PARTICLE_VELOCITY_SCALE * (1 + self.freq_data.get("highs", 0.0) * 2)
            p["y"] += p["vy"] * PARTICLE_VELOCITY_SCALE * (1 + self.freq_data.get("highs", 0.0) * 2)
            
            # Apply bass impulse (e.g., push particles out)
            bass_impulse_strength = self.freq_data.get("bass", 0.0) * 5 # Scale for effect
            if bass_impulse_strength > 0.1: # Only if significant bass
                dx = p["x"] - cx
                dy = p["y"] - cy
                dist = np.sqrt(dx**2 + dy**2) + 1e-8
                p["x"] += (dx / dist) * bass_impulse_strength
                p["y"] += (dy / dist) * bass_impulse_strength

            p["life"] -= 1
            # Fade particles
            p["alpha"] = int(255 * (p["life"] / 300.0) * (1 + self.freq_data.get("mids", 0.0))) # Mids affect density/visibility

            # Redraw if alive and visible
            if p["life"] > 0 and p["alpha"] > 0:
                color_with_alpha = (self.orb_color[0], self.orb_color[1], self.orb_color[2], p["alpha"])
                particle_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color_with_alpha, (p["size"], p["size"]), int(p["size"]))
                self.screen.blit(particle_surf, (int(p["x"] - p["size"]), int(p["y"] - p["size"])))
                new_particles.append(p)
            else:
                new_particles.append(self._create_particle()) # Re-spawn
        self.particles = new_particles

        # --- Scrolling Chord Timeline ---
        text_y = cy + ORB_BASE_RADIUS + ORB_MAX_RADIUS_ADD + 50 # Position below orb

        # Filter out future/past chords for display window
        display_chords: List[Dict[str, Any]] = []
        for i, chord_data in enumerate(self.chord_timeline):
            if chord_data['timestamp'] >= (self.playback_position_ms - CHORD_TEXT_FADE_DISTANCE * 1000 / (self.width()/100)): # Heuristic based on screen width
                display_chords.append(chord_data)
            if len(display_chords) > 0 and chord_data['timestamp'] > (self.playback_position_ms + self.width() * 1000 / 100): # Stop far right
                break
        
        # Find current chord index to center it
        current_chord_idx = -1
        for i in range(len(self.chord_timeline) - 1, -1, -1):
            if self.chord_timeline[i]['timestamp'] <= self.playback_position_ms:
                current_chord_idx = i
                break
        
        if current_chord_idx != -1:
            # Calculate a scroll offset so the current chord is centered
            # This is complex as it depends on chord text width
            # We will approximate this by simply rendering around the center
            
            # Draw scrolling chords around the current position

            # Iterate through chords around the current position
            # We'll render chords in a fixed visible range relative to the current playback position
            # and let them scroll/fade based on their timestamp.

            # Determine visible range of timestamps (unused but kept for clarity)

            # Go left from current chord
            for i in range(current_chord_idx, -1, -1):
                chord_data = self.chord_timeline[i]
                chord_time_ms = chord_data['timestamp']
                chord_name = chord_data['chord']
                base_color = CHORD_COLORS.get(chord_name, CHORD_COLORS["Unknown"])

                # Calculate position relative to center
                pos_diff_ms = chord_time_ms - self.playback_position_ms
                render_x = cx + (pos_diff_ms * (150 / 1000)) # 150 pixels per second scroll speed

                # Check if it's within display bounds
                if render_x + self.chord_font.size(chord_name)[0] < 0 or render_x > self.width():
                    continue # Too far off screen

                # Calculate alpha based on distance from center (or current timestamp)
                distance_from_center_px = abs(render_x - cx)
                alpha = 255
                if distance_from_center_px > self.width() / 4: # Start fading if outside central quarter
                    fade_progress = (distance_from_center_px - self.width() / 4) / (self.width() / 4)
                    alpha = int(255 * (1 - min(1.0, fade_progress)))

                if alpha > 0:
                    text_surface = self.chord_font.render(chord_name, True, base_color)
                    text_surface.set_alpha(alpha)
                    text_rect = text_surface.get_rect(center=(render_x, text_y))
                    self.screen.blit(text_surface, text_rect)
            
            # Go right from current chord
            for i in range(current_chord_idx + 1, len(self.chord_timeline)):
                chord_data = self.chord_timeline[i]
                chord_time_ms = chord_data['timestamp']
                chord_name = chord_data['chord']
                base_color = CHORD_COLORS.get(chord_name, CHORD_COLORS["Unknown"])

                pos_diff_ms = chord_time_ms - self.playback_position_ms
                render_x = cx + (pos_diff_ms * (150 / 1000))

                if render_x < 0 or render_x - self.chord_font.size(chord_name)[0] > self.width():
                    continue

                distance_from_center_px = abs(render_x - cx)
                alpha = 255
                if distance_from_center_px > self.width() / 4:
                    fade_progress = (distance_from_center_px - self.width() / 4) / (self.width() / 4)
                    alpha = int(255 * (1 - min(1.0, fade_progress)))
                
                if alpha > 0:
                    text_surface = self.chord_font.render(chord_name, True, base_color)
                    text_surface.set_alpha(alpha)
                    text_rect = text_surface.get_rect(center=(render_x, text_y))
                    self.screen.blit(text_surface, text_rect)
            
            # Draw current chord prominently
            current_chord_text_surface = self.orb_font.render(self.current_chord_name, True, self.current_orb_color)
            current_chord_text_rect = current_chord_text_surface.get_rect(center=(cx, cy + ORB_BASE_RADIUS + ORB_MAX_RADIUS_ADD * 0.5 + 10)) # Adjust pos
            self.screen.blit(current_chord_text_surface, current_chord_text_rect)

        # Update the entire screen
        pygame.display.flip()

    def update_qt(self):
        # This forces Qt to re-paint this widget, which in turn calls paintEvent
        # and then update_visuals. This ensures Qt and Pygame are somewhat synchronized.
        # However, the QTimer already calls update_visuals directly.
        # This method is mostly for when Pygame would need to be "embedded" more traditionally.
        # For this setup, the direct call to update_visuals from QTimer is sufficient.
        pass


# --- Main PyQt6 Application Window ---
class ChordPlayerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Code Maniac's Chord Player (Visualizer Edition)")
        self.setGeometry(100, 100, 600, 300) # Smaller, control-focused window
        
        self.audio_processor = LiveAudioProcessor()
        self.audio_processor.volume_data.connect(self._handle_volume_data)
        self.audio_processor.freq_band_data.connect(self._handle_freq_data)
        self.audio_processor.position_updated.connect(self.update_playback_progress)
        self.audio_processor.playback_finished.connect(self.handle_playback_finished)
        self.audio_processor.start() # Start the audio processing thread

        self.chord_detector_thread = ChordDetector()
        self.chord_detector_thread.chords_detected.connect(self.handle_chords_detected)
        self.chord_detector_thread.detection_error.connect(self.handle_detection_error)
        self.chord_detector_thread.detection_progress.connect(self.update_detection_progress)
        self.chord_detector_thread.start() # Start chord detection thread

        self.visualizer = VisualizerWindow()
        self.visualizer.window_closed.connect(self._handle_visualizer_closed) # Connect signal
        self.visualizer.show()

        self.current_song_path: Optional[str] = None
        self.current_chord_data: List[Dict[str, Any]] = []
        self.song_duration_ms: int = 0
        self.is_playing: bool = False
        self.is_paused: bool = False

        self.init_ui()
        self.update_button_states()

    def init_ui(self):
        # Apply dark theme
        self.set_dark_theme()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Song Info & Status
        self.song_info_label = QLabel("No song loaded.")
        self.song_info_label.setFont(QFont("Arial", 12))
        self.song_info_label.setStyleSheet("color: #ecf0f1;")
        main_layout.addWidget(self.song_info_label)

        # Chord Detection Progress
        self.detection_progress_dialog = QProgressDialog("Detecting Chords...", "Abort", 0, 100, self)
        self.detection_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.detection_progress_dialog.setWindowTitle("Analyzing Audio")
        self.detection_progress_dialog.setMinimumDuration(0)
        self.detection_progress_dialog.setValue(0)
        self.detection_progress_dialog.hide()
        self.detection_progress_dialog.canceled.connect(self._cancel_detection)

        # Playback Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        self.load_song_button = QPushButton("Load Song")
        self.load_song_button.clicked.connect(self.load_song_dialog)
        self.load_song_button.setFixedSize(100, 35)
        self.apply_button_style(self.load_song_button, "#9b59b6")

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_pause_song)
        self.play_button.setFixedSize(80, 35)
        self.apply_button_style(self.play_button, "#2ecc71")

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_song)
        self.stop_button.setFixedSize(80, 35)
        self.apply_button_style(self.stop_button, "#e74c3c")

        controls_layout.addWidget(self.load_song_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Seek Slider
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self._seek_slider_moved)
        self.position_slider.sliderReleased.connect(self._seek_slider_released)
        self.apply_slider_style(self.position_slider, "#1abc9c")
        main_layout.addWidget(self.position_slider)

        # Time Display
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setFont(QFont("Monospace", 10))
        self.time_label.setStyleSheet("color: #bdc3c7;")
        main_layout.addWidget(self.time_label)

        # Volume Control
        volume_layout = QHBoxLayout()
        volume_layout.addStretch()
        volume_label = QLabel("Volume:")
        volume_label.setFont(QFont("Arial", 10))
        volume_label.setStyleSheet("color: #bdc3c7;")
        volume_layout.addWidget(volume_label)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70) # Default volume
        self.volume_slider.setFixedWidth(150)
        self.volume_slider.valueChanged.connect(self.change_volume)
        self.apply_slider_style(self.volume_slider, "#3498db")
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addStretch()
        main_layout.addLayout(volume_layout)
        
        self.change_volume(self.volume_slider.value()) # Set initial volume
    
    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(52, 73, 94)) # Dark Blue-Grey
        palette.setColor(QPalette.ColorRole.WindowText, QColor(236, 240, 241)) # Off-White
        palette.setColor(QPalette.ColorRole.Base, QColor(44, 62, 80)) # Even Darker Blue-Grey
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(39, 55, 70))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(236, 240, 241))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(44, 62, 80))
        palette.setColor(QPalette.ColorRole.Text, QColor(236, 240, 241))
        palette.setColor(QPalette.ColorRole.Button, QColor(64, 84, 104))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(236, 240, 241))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(52, 152, 219))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(52, 152, 219)) # Blue for highlight
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        QApplication.instance().setPalette(palette) # Apply to whole app

    def apply_button_style(self, button: QPushButton, color: str):
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 5px;
                font-size: 13px;
                padding: 8px;
            }}
            QPushButton:hover {{ background-color: {color[:-2]}B3; }} # Simple hover effect
            QPushButton:pressed {{ background-color: {color[:-2]}80; }} # Pressed effect
            QPushButton:disabled {{ background-color: #555; color: #aaa; }}
        """)

    def apply_slider_style(self, slider: QSlider, color: str):
        slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid #555;
                height: 6px;
                background: #444;
                margin: 2px 0;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {color};
                border: 1px solid {color};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {color};
                border-radius: 3px;
            }}
        """)

    def update_button_states(self):
        has_song = self.current_song_path is not None
        
        self.play_button.setEnabled(has_song)
        if self.is_playing:
            self.play_button.setText("Pause")
            self.stop_button.setEnabled(True)
        elif self.is_paused:
            self.play_button.setText("Resume")
            self.stop_button.setEnabled(True)
        else:
            self.play_button.setText("Play")
            self.stop_button.setEnabled(False)
        self.position_slider.setEnabled(has_song)


    def load_song_dialog(self):
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Audio File", "",
            "Audio Files (*.mp3 *.wav *.ogg *.flac *.m4a);;All Files (*)", options=options
        )
        if file_name:
            self.current_song_path = file_name
            self.stop_song() # Stop any current playback or analysis

            self.song_info_label.setText(f"Loading: {os.path.basename(file_name)}")
            
            # Start chord detection
            self.chord_detector_thread.set_audio_file(file_name)
            # Restart the thread if it was already finished (QThread cannot be started twice)
            if self.chord_detector_thread.isRunning():
                self.chord_detector_thread.terminate() # Force stop current detection if any
                self.chord_detector_thread.wait() # Wait for it to finish
                self.chord_detector_thread = ChordDetector() # Create new instance
                self.chord_detector_thread.chords_detected.connect(self.handle_chords_detected)
                self.chord_detector_thread.detection_error.connect(self.handle_detection_error)
                self.chord_detector_thread.detection_progress.connect(self.update_detection_progress)
            self.chord_detector_thread.start()

            self.detection_progress_dialog.setValue(0)
            self.detection_progress_dialog.show()
            self.update_button_states()


    def _cancel_detection(self):
        if self.chord_detector_thread.isRunning():
            self.chord_detector_thread.terminate() # Request termination
            self.chord_detector_thread.wait() # Wait for thread to finish
        self.detection_progress_dialog.hide()
        self.current_song_path = None
        self.song_info_label.setText("Analysis cancelled. No song loaded.")
        QMessageBox.information(self, "Cancelled", "Chord detection cancelled.")
        self.update_button_states()

    def handle_chords_detected(self, chords_list: List[Dict[str, Any]], duration_seconds: float):
        self.detection_progress_dialog.hide()
        self.current_chord_data = chords_list
        self.song_duration_ms = int(duration_seconds * 1000)

        if not self.audio_processor.load_song(self.current_song_path):
            QMessageBox.critical(self, "Audio Load Error", "Could not load audio for playback. Check file format or system audio drivers.")
            self.current_song_path = None
            self.song_info_label.setText("Load failed. No song loaded.")
            self.song_duration_ms = 0
            self.current_chord_data = []
            self.visualizer.set_chord_timeline([], 0)
        else:
            self.song_info_label.setText(f"Ready: {os.path.basename(self.current_song_path)}")
            self.position_slider.setRange(0, self.song_duration_ms)
            self.update_time_label(0, self.song_duration_ms)
            self.visualizer.set_chord_timeline(self.current_chord_data, self.song_duration_ms)
            self.play_song() # Auto-play after detection

        self.update_button_states()

    def update_detection_progress(self, value: int):
        self.detection_progress_dialog.setValue(value)
        # Check if dialog was closed by user
        if self.detection_progress_dialog.wasCanceled():
            self._cancel_detection()


    def handle_detection_error(self, error_message: str):
        self.detection_progress_dialog.hide()
        QMessageBox.critical(self, "Chord Detection Error", error_message)
        self.current_song_path = None
        self.song_info_label.setText("No song loaded.")
        self.song_duration_ms = 0
        self.current_chord_data = []
        self.visualizer.set_chord_timeline([], 0)
        self.update_button_states()

    def play_pause_song(self):
        if not self.current_song_path:
            QMessageBox.warning(self, "No Song", "Please load an audio file first.")
            return

        if self.is_playing:
            self.audio_processor.pause()
            self.is_playing = False
            self.is_paused = True
        else: # Not playing, either stopped or paused
            self.audio_processor.play()
            self.is_playing = True
            self.is_paused = False
        self.update_button_states()

    def play_song(self):
        if not self.is_playing:
            self.audio_processor.play()
            self.is_playing = True
            self.is_paused = False
            self.update_button_states()

    def stop_song(self):
        self.audio_processor.stop()
        self.is_playing = False
        self.is_paused = False
        self.update_button_states()
        self.visualizer.set_realtime_data(0.0, {"bass": 0.0, "mids": 0.0, "highs": 0.0}, 0) # Reset visualizer
        self.update_time_label(0, self.song_duration_ms)
        self.position_slider.setValue(0) # Reset slider


    def change_volume(self, value: int):
        volume_level = float(value) / 100.0
        sd.set_output_latency('low') # Try to ensure low latency
        sd.default.samplerate = AUDIO_SR # Set samplerate globally for sounddevice
        sd.default.channels = 1 # Mono output
        sd.default.dtype = 'float32'
        sd.default.blocksize = BUFFER_SIZE
        sd.default.volume = volume_level # This sets global volume for sounddevice
        # Note: sd.default.volume affects all sounddevice streams.
        # For per-stream volume, you'd modify audio data directly in the callback.


    def _seek_slider_moved(self, value: int):
        # Pause real-time updates while dragging, but update the visualizer
        volume = getattr(self.audio_processor.volume_data, "current_value", 0.0)
        freq = getattr(self.audio_processor.freq_band_data, "current_value", {"bass": 0.0, "mids": 0.0, "highs": 0.0})
        self.visualizer.set_realtime_data(volume, freq, value)  # Use last known volume/freq
        self.update_time_label(value, self.song_duration_ms)
        self.is_playing = self.audio_processor._is_playing # Remember current state
        self.audio_processor.pause() # Temporarily pause real stream to avoid jitter


    def _seek_slider_released(self):
        target_ms = self.position_slider.value()
        self.audio_processor.set_position(target_ms)
        if self.is_playing: # Resume if it was playing before drag
            self.audio_processor.play()
        self.update_button_states() # Update button state

    def update_playback_progress(self, position_ms: int):
        if not self.position_slider.isSliderDown(): # Only update slider if user is not dragging
            self.position_slider.setValue(position_ms)
        self.update_time_label(position_ms, self.song_duration_ms)

        # Pass current state to visualizer
        self.visualizer.set_realtime_data(
            self.audio_processor.volume_data.current_value, # Access last emitted value
            self.audio_processor.freq_band_data.current_value, # Access last emitted value
            position_ms
        )

    def update_time_label(self, current_ms: int, total_ms: int):
        current_s = current_ms // 1000
        total_s = total_ms // 1000

        min_current = current_s // 60
        sec_current = current_s % 60

        min_total = total_s // 60
        sec_total = total_s % 60

        self.time_label.setText(
            f"{min_current:02d}:{sec_current:02d} / {min_total:02d}:{sec_total:02d}"
        )

    def handle_playback_finished(self):
        print("Playback finished signal received (from audio processor).")
        self.stop_song() # Reset UI


    def _handle_volume_data(self, volume: float):
        # We store the last emitted value for access by update_playback_progress
        self.audio_processor.volume_data.current_value = volume

    def _handle_freq_data(self, freq_data: Dict[str, float]):
        # We store the last emitted value for access by update_playback_progress
        self.audio_processor.freq_band_data.current_value = freq_data

    def _handle_visualizer_closed(self):
        # If user closes visualizer window, also close main app
        self.close()

    def closeEvent(self, event: Any): # QCloseEvent
        print("Application closing...")
        
        # Stop and quit all threads
        self.audio_processor.stop() # Ensure stream is closed
        self.audio_processor.quit()
        self.audio_processor.wait(5000) # Give time to finish

        self.chord_detector_thread.terminate() # Force stop if still running analysis
        self.chord_detector_thread.wait(5000)
        
        # Close Pygame window first
        if self.visualizer:
            self.visualizer.close() # Emit window_closed signal

        pygame.quit() # Ensure Pygame resources are released
        sd.stop() # Stop any remaining sounddevice streams
        sd.terminate() # Terminate sounddevice (important for PortAudio release)

        super().closeEvent(event)

if __name__ == "__main__":
    # Monkey-patch signals to store their last emitted value for easier access
    # This is a hack, proper way is to use a shared state object or a QProperty
    def _monkey_patch_signal(signal_obj):
        old_emit = signal_obj.emit
        signal_obj.current_value = None
        def new_emit(*args, **kwargs):
            if args:
                signal_obj.current_value = args[0] # Store the first argument
            old_emit(*args, **kwargs)
        signal_obj.emit = new_emit
        return signal_obj

    _monkey_patch_signal(LiveAudioProcessor.volume_data)
    _monkey_patch_signal(LiveAudioProcessor.freq_band_data)

    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Modern look

    player = ChordPlayerApp()
    player.show()
    sys.exit(app.exec())

