#!/usr/bin/env python3
"""
WAV Keypress Detector
Load một file WAV và đánh dấu các điểm có thể là chỗ đã bấm phím.
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
import argparse
import os
import json

class WavKeypressDetector:
    def __init__(self, threshold_multiplier=2.0, min_distance=0.1):
        """
        Initialize WAV keypress detector.
        
        Args:
            threshold_multiplier: Multiplier for threshold calculation (default: 2.0)
            min_distance: Minimum distance between keypress events in seconds (default: 0.1)
        """
        self.threshold_multiplier = threshold_multiplier
        self.min_distance = min_distance
        self.audio_data = None
        self.sample_rate = None
        self.timestamps = None
        self.rms_values = None
        
    def load_wav_file(self, file_path):
        """
        Load WAV file and prepare audio data.
        
        Args:
            file_path: Path to the WAV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load audio file
            self.audio_data, self.sample_rate = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(self.audio_data.shape) > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
            
            # Create timestamps
            duration = len(self.audio_data) / self.sample_rate
            self.timestamps = np.linspace(0, duration, len(self.audio_data))
            
            print(f"✅ Loaded WAV file: {file_path}")
            print(f"📊 Duration: {duration:.2f} seconds")
            print(f"🎵 Sample rate: {self.sample_rate} Hz")
            print(f"📈 Samples: {len(self.audio_data)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading WAV file: {e}")
            return False
    
    def calculate_rms(self, window_size=1024):
        """
        Calculate RMS (Root Mean Square) values for audio data.
        
        Args:
            window_size: Size of the sliding window for RMS calculation
        """
        if self.audio_data is None:
            raise ValueError("No audio data loaded")
        
        # Calculate RMS values using sliding window
        rms_values = []
        rms_timestamps = []
        
        for i in range(0, len(self.audio_data) - window_size, window_size // 4):
            window = self.audio_data[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
            rms_timestamps.append(self.timestamps[i + window_size // 2])
        
        self.rms_values = np.array(rms_values)
        self.rms_timestamps = np.array(rms_timestamps)
        
        print(f"📊 Calculated RMS values: {len(self.rms_values)} points")
    
    def detect_keypress_events(self, smoothing_window=11):
        """
        Detect potential keypress events based on audio analysis.
        
        Args:
            smoothing_window: Window size for smoothing filter (must be odd)
            
        Returns:
            tuple: (keypress_times, keypress_values) - times and RMS values of detected keypresses
        """
        if self.rms_values is None:
            raise ValueError("RMS values not calculated")
        
        # Smooth the RMS values to reduce noise
        if len(self.rms_values) >= smoothing_window:
            smoothed_rms = savgol_filter(self.rms_values, smoothing_window, 3)
        else:
            smoothed_rms = self.rms_values
        
        # Calculate dynamic threshold
        baseline_level = np.percentile(smoothed_rms, 25)  # 25th percentile as baseline
        noise_level = np.std(smoothed_rms[smoothed_rms <= baseline_level])
        threshold = baseline_level + (noise_level * self.threshold_multiplier)
        
        # Find peaks above threshold
        min_samples_distance = int(self.min_distance * self.sample_rate / (1024 // 4))
        peaks, properties = find_peaks(
            smoothed_rms, 
            height=threshold,
            distance=min_samples_distance,
            prominence=noise_level * 0.5
        )
        
        keypress_times = self.rms_timestamps[peaks]
        keypress_values = smoothed_rms[peaks]
        
        print(f"🔍 Detection results:")
        print(f"   📊 Baseline level: {baseline_level:.6f}")
        print(f"   📊 Noise level: {noise_level:.6f}")
        print(f"   🎯 Threshold: {threshold:.6f}")
        print(f"   ⌨️  Detected keypresses: {len(keypress_times)}")
        
        return keypress_times, keypress_values, threshold, smoothed_rms
    
    def visualize_results(self, keypress_times, keypress_values, threshold, smoothed_rms, save_plot=True):
        """
        Visualize the results with detected keypress events.
        
        Args:
            keypress_times: Times of detected keypresses
            keypress_values: RMS values at keypress times
            threshold: Detection threshold
            smoothed_rms: Smoothed RMS values
            save_plot: Whether to save the plot as an image
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Raw audio waveform
        plt.subplot(3, 1, 1)
        plt.plot(self.timestamps, self.audio_data, alpha=0.7, color='blue')
        plt.title('Raw Audio Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for detected keypresses
        for kp_time in keypress_times:
            plt.axvline(x=kp_time, color='red', linestyle='--', alpha=0.7)
        
        # Plot 2: RMS values with detection
        plt.subplot(3, 1, 2)
        plt.plot(self.rms_timestamps, self.rms_values, alpha=0.7, color='green', label='Raw RMS')
        plt.plot(self.rms_timestamps, smoothed_rms, color='darkgreen', linewidth=2, label='Smoothed RMS')
        plt.axhline(y=threshold, color='orange', linestyle='-', linewidth=2, label=f'Threshold: {threshold:.6f}')
        
        # Mark detected keypresses
        plt.scatter(keypress_times, keypress_values, color='red', s=100, zorder=5, 
                   label=f'Detected Keypresses ({len(keypress_times)})')
        
        plt.title('RMS Analysis with Keypress Detection')
        plt.xlabel('Time (seconds)')
        plt.ylabel('RMS Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Keypress events timeline
        plt.subplot(3, 1, 3)
        plt.eventplot(keypress_times, colors='red', lineoffsets=1, linelengths=0.8)
        plt.title('Keypress Events Timeline')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Events')
        plt.grid(True, alpha=0.3)
        plt.ylim(0.5, 1.5)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keypress_detection_results_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📁 Plot saved as: {filename}")
        
        plt.show()
    
    def export_results(self, keypress_times, filename=None):
        """
        Export detected keypress times to a JSON file.
        
        Args:
            keypress_times: Array of detected keypress times
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keypress_times_{timestamp}.json"
        
        # Convert numpy array to list for JSON serialization
        keypress_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_keypresses": len(keypress_times),
            "keypress_times": keypress_times.tolist(),
            "analysis_parameters": {
                "threshold_multiplier": self.threshold_multiplier,
                "min_distance": self.min_distance
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(keypress_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Results exported to JSON: {filename}")
    
    def analyze_wav_file(self, file_path, window_size=1024, smoothing_window=11, 
                        visualize=True, export=True):
        """
        Complete analysis workflow for a WAV file.
        
        Args:
            file_path: Path to the WAV file
            window_size: RMS calculation window size
            smoothing_window: Smoothing filter window size
            visualize: Whether to show visualization
            export: Whether to export results
            
        Returns:
            tuple: (keypress_times, keypress_values) or None if failed
        """
        print(f"🎵 Analyzing WAV file: {file_path}")
        print("=" * 60)
        
        # Load WAV file
        if not self.load_wav_file(file_path):
            return None
        
        # Calculate RMS values
        self.calculate_rms(window_size)
        
        # Detect keypress events
        keypress_times, keypress_values, threshold, smoothed_rms = self.detect_keypress_events(smoothing_window)
        
        # Visualize results
        if visualize:
            self.visualize_results(keypress_times, keypress_values, threshold, smoothed_rms)
        
        # Export results
        if export:
            self.export_results(keypress_times)
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Summary:")
        print(f"   - File duration: {len(self.audio_data) / self.sample_rate:.2f} seconds")
        print(f"   - Detected keypresses: {len(keypress_times)}")
        print(f"   - Average keypress interval: {np.mean(np.diff(keypress_times)):.2f} seconds" if len(keypress_times) > 1 else "")
        
        return keypress_times, keypress_values


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="WAV Keypress Detector")
    parser.add_argument("wav_file", help="Path to WAV file to analyze")
    parser.add_argument("--threshold", type=float, default=2.0, 
                       help="Threshold multiplier for detection (default: 2.0)")
    parser.add_argument("--min-distance", type=float, default=0.1,
                       help="Minimum distance between keypresses in seconds (default: 0.1)")
    parser.add_argument("--window-size", type=int, default=1024,
                       help="RMS calculation window size (default: 1024)")
    parser.add_argument("--smoothing-window", type=int, default=11,
                       help="Smoothing filter window size (default: 11)")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Don't show visualization")
    parser.add_argument("--no-export", action="store_true",
                       help="Don't export results to file")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.wav_file):
        print(f"❌ Error: File '{args.wav_file}' not found")
        return
    
    # Create detector
    detector = WavKeypressDetector(
        threshold_multiplier=args.threshold,
        min_distance=args.min_distance
    )
    
    # Analyze file
    result = detector.analyze_wav_file(
        args.wav_file,
        window_size=args.window_size,
        smoothing_window=args.smoothing_window,
        visualize=not args.no_visualize,
        export=not args.no_export
    )
    
    if result is None:
        print("❌ Analysis failed")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
