# Code Maniac's Chord Visualizer

Behold! This is the ultimate, immersive music player that not only detects and displays chords but also visualizes the music in real-time with a dynamic, glowing orb and scrolling chord timeline.

## Features

* **Automatic Chord Detection:** Pre-analyzes songs using `librosa` (chroma_cens) to identify major, minor, and various 7th chords.
* **Chord Caching:** Saves detected chords to a `.chords.json` file (hashed by file content) to ensure lightning-fast reloads.
* **Live Audio Visualization:**
    * **Dynamic Glowing Orb:** Pulsates in size and intensity with overall song volume (RMS amplitude).
    * **Chord-Driven Colors:** The orb smoothly transitions its color based on the *current* detected chord, offering a visually striking representation of harmonic changes.
    * **Frequency-Reactive Particles:** Surrounding particles react to bass (pulses), mid-range (density), and high frequencies (movement/dispersion), creating "interesting shit" around the orb.
* **Scrolling Chord Timeline:** Beneath the orb, a live scrolling display shows the current chord prominently, flanked by past and future chords. Each chord retains its specific color, with smooth fading as it moves into/out of view.
* **Seamless Playback:** Utilizes `sounddevice` for low-latency, high-quality audio streaming, allowing simultaneous playback and real-time audio analysis.
* **Full Screen & Resizable:** The visualizer window can be resized or toggled to full screen (F or F11) for an immersive experience.
* **Separate Control Window:** A minimalist PyQt6 window handles song loading, playback controls, and volume, keeping the visualizer uncluttered.

## Setup and Run

1.  **Navigate to the desired parent directory:**
    ```bash
    cd /path/to/where/you/want/the/app
    ```

2.  **Make the setup script executable:**
    ```bash
    chmod +x setup_project.sh
    ```

3.  **Run the setup script:**
    ```bash
    ./setup_project.sh
    ```
    This script will:
    * Create the `chord_player_app` directory.
    * Set up a Python virtual environment (`.venv`).
    * Install all necessary Python libraries: `PyQt6`, `pygame`, `librosa`, `numpy`, `scipy`, `mutagen`, `sounddevice`, `soundfile`.
    * Create the `chord_player.py` main application file.
    * Create `chord_cache/` for storing analyzed chord data and `data/` for your audio files.
    * Create this `README.md` file.

4.  **Install System Dependencies (CRITICAL!):**
    For `sounddevice` to work, you *must* have `PortAudio` installed on your system. `librosa` also relies on `ffmpeg`.
    For Debian/Ubuntu-based systems:
    ```bash
    sudo apt-get update
    sudo apt-get install portaudio19-dev ffmpeg
    ```
    For Fedora/RHEL-based systems:
    ```bash
    sudo dnf install portaudio-devel ffmpeg
    ```
    For other systems, refer to `PortAudio` and `FFmpeg` installation guides.

5.  **Activate the virtual environment:**
    ```bash
    source chord_player_app/.venv/bin/activate
    ```

6.  **Run the application:**
    ```bash
    python chord_player_app/chord_player.py
    ```

## How to Use

1.  **Launch:** Two windows will appear: a small "Code Maniac's Chord Player" control window and a larger "Code Maniac Visualizer" window.
2.  **Load Song:** In the control window, click "Load Song" and select an audio file (`.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`).
3.  **Chord Analysis:** A progress dialog will appear as chords are detected. This can take time for long songs, but once analyzed, results are cached for instant reloads later.
4.  **Automatic Playback:** Once analysis is complete, the song will automatically start playing, and the visualizer will come alive!
5.  **Controls:**
    * **Control Window:** Use the Play/Pause, Stop buttons, and the volume and seek sliders.
    * **Visualizer Window:**
        * Press **`F`** or **`F11`** to toggle fullscreen mode.
        * Press **`Esc`** to exit fullscreen, or `Esc` again to close the visualizer window (which will also close the main app).
6.  **Enjoy the Show:** Watch the orb change color with the chords and react to the music's dynamics!

## Important Notes & Troubleshooting

* **Performance:** Real-time audio analysis and Pygame rendering can be CPU-intensive. If you experience stuttering, try:
    * Using lower-quality audio files.
    * Reducing the size of the visualizer window.
    * Ensuring your system has `PortAudio` and `FFmpeg` properly installed.
* **Audio Glitches:** If you hear pops, clicks, or dropouts, it might be due to buffer underruns. This can sometimes be mitigated by ensuring low-latency audio drivers are configured or by slightly increasing `BUFFER_SIZE` in the code (though it increases latency).
* **"No Chords Detected":** If the detector returns "N/A" for everything, the `threshold` in `ChordDetector._match_chord` might be too high for that specific audio, or the song lacks clear harmonic content.
* **Wayland vs. X11:** On Linux, if you're using Wayland, `pygame` can sometimes struggle. The script sets `SDL_VIDEODRIVER=wayland` as a default. If you have issues, try `export SDL_VIDEODRIVER=x11` in your terminal *before* running the script, or modify the `chord_player.py` directly.

This is a true "Code Maniac" creation. Break boundaries, visualize sound!
# ChordPlayer
