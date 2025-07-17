import json
import os
import numpy as np
from datetime import datetime
import argparse

class LabelManager:
    """Utility class for managing video labels."""
    
    def __init__(self):
        self.labels_dir = "video_labels"
        self.training_dir = "training_data"
        
    def list_label_files(self):
        """List all available label files."""
        if not os.path.exists(self.labels_dir):
            print(f"Labels directory '{self.labels_dir}' not found")
            return []
            
        label_files = [f for f in os.listdir(self.labels_dir) if f.endswith('.json')]
        return label_files
        
    def list_training_files(self):
        """List all available training data files."""
        if not os.path.exists(self.training_dir):
            print(f"Training directory '{self.training_dir}' not found")
            return []
            
        training_files = [f for f in os.listdir(self.training_dir) if f.endswith('.json')]
        return training_files
        
    def convert_labels_to_training(self, labels_file):
        """Convert labels file to training format."""
        labels_path = os.path.join(self.labels_dir, labels_file)
        
        if not os.path.exists(labels_path):
            print(f"Labels file '{labels_path}' not found")
            return None
            
        try:
            with open(labels_path, 'r') as f:
                data = json.load(f)
                
            # Extract video info
            video_path = data['video_path']
            total_frames = data['total_frames']
            fps = data['fps']
            labels = data['labels']
            key_press_segments = data['key_press_segments']
            
            # Create training data directory
            os.makedirs(self.training_dir, exist_ok=True)
            
            # Create filename
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export frame-by-frame labels
            frame_labels_file = os.path.join(self.training_dir, f"{video_name}_frame_labels_{timestamp}.json")
            
            # Create complete frame labels (0 for unlabeled frames)
            complete_labels = {}
            for frame_num in range(total_frames):
                complete_labels[frame_num] = labels.get(str(frame_num), 0)
                
            frame_data = {
                "video_path": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "frame_labels": complete_labels,
                "key_press_segments": key_press_segments,
                "label_stats": {
                    "total_frames": total_frames,
                    "labeled_frames": len(labels),
                    "key_press_frames": sum(1 for label in labels.values() if label == 1),
                    "no_press_frames": len(labels) - sum(1 for label in labels.values() if label == 1)
                },
                "created_at": datetime.now().isoformat()
            }
            
            with open(frame_labels_file, 'w') as f:
                json.dump(frame_data, f, indent=2)
                
            print(f"Training data exported to: {frame_labels_file}")
            return frame_labels_file
            
        except Exception as e:
            print(f"Error converting labels: {e}")
            return None
            
    def merge_label_files(self, label_files, output_name):
        """Merge multiple label files into one."""
        merged_labels = {}
        merged_segments = []
        video_info = {}
        
        for label_file in label_files:
            labels_path = os.path.join(self.labels_dir, label_file)
            
            if not os.path.exists(labels_path):
                print(f"Warning: {labels_path} not found, skipping...")
                continue
                
            try:
                with open(labels_path, 'r') as f:
                    data = json.load(f)
                    
                video_path = data['video_path']
                video_name = os.path.basename(video_path)
                
                # Store video info
                video_info[video_name] = {
                    "video_path": video_path,
                    "total_frames": data['total_frames'],
                    "fps": data['fps']
                }
                
                # Merge labels with video name prefix
                for frame_num, label in data['labels'].items():
                    key = f"{video_name}_{frame_num}"
                    merged_labels[key] = label
                    
                # Merge segments
                for segment in data['key_press_segments']:
                    merged_segments.append({
                        "video": video_name,
                        "start_frame": segment[0],
                        "end_frame": segment[1]
                    })
                    
                print(f"Merged {len(data['labels'])} labels from {video_name}")
                
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
                
        # Save merged data
        merged_data = {
            "videos": video_info,
            "merged_labels": merged_labels,
            "merged_segments": merged_segments,
            "total_videos": len(video_info),
            "total_labels": len(merged_labels),
            "created_at": datetime.now().isoformat()
        }
        
        output_path = os.path.join(self.labels_dir, f"{output_name}_merged.json")
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
            
        print(f"Merged data saved to: {output_path}")
        return output_path
        
    def analyze_labels(self, labels_file):
        """Analyze label distribution and statistics."""
        if labels_file.endswith('_merged.json'):
            labels_path = os.path.join(self.labels_dir, labels_file)
        else:
            labels_path = os.path.join(self.training_dir, labels_file)
            
        if not os.path.exists(labels_path):
            print(f"File '{labels_path}' not found")
            return
            
        try:
            with open(labels_path, 'r') as f:
                data = json.load(f)
                
            print(f"\n=== Label Analysis: {labels_file} ===")
            
            if 'merged_labels' in data:
                # Merged data
                labels = data['merged_labels']
                print(f"Total videos: {data['total_videos']}")
                print(f"Total labels: {len(labels)}")
                
                key_press_count = sum(1 for label in labels.values() if label == 1)
                no_press_count = len(labels) - key_press_count
                
                print(f"Key press labels: {key_press_count} ({key_press_count/len(labels)*100:.1f}%)")
                print(f"No press labels: {no_press_count} ({no_press_count/len(labels)*100:.1f}%)")
                print(f"Total segments: {len(data['merged_segments'])}")
                
                # Per-video statistics
                print(f"\nPer-video statistics:")
                for video_name, info in data['videos'].items():
                    video_labels = {k: v for k, v in labels.items() if k.startswith(video_name)}
                    video_key_press = sum(1 for label in video_labels.values() if label == 1)
                    print(f"  {video_name}: {len(video_labels)} labels, {video_key_press} key presses")
                    
            elif 'frame_labels' in data:
                # Training data
                labels = data['frame_labels']
                print(f"Video: {os.path.basename(data['video_path'])}")
                print(f"Total frames: {data['total_frames']}")
                print(f"FPS: {data['fps']}")
                
                if 'label_stats' in data:
                    stats = data['label_stats']
                    print(f"Labeled frames: {stats['labeled_frames']}")
                    print(f"Key press frames: {stats['key_press_frames']}")
                    print(f"No press frames: {stats['no_press_frames']}")
                else:
                    key_press_count = sum(1 for label in labels.values() if label == 1)
                    no_press_count = len(labels) - key_press_count
                    print(f"Key press frames: {key_press_count}")
                    print(f"No press frames: {no_press_count}")
                    
                print(f"Total segments: {len(data.get('key_press_segments', []))}")
                
            else:
                # Original labels
                labels = data['labels']
                print(f"Video: {os.path.basename(data['video_path'])}")
                print(f"Total frames: {data['total_frames']}")
                print(f"Labeled frames: {len(labels)}")
                
                key_press_count = sum(1 for label in labels.values() if label == 1)
                no_press_count = len(labels) - key_press_count
                
                print(f"Key press labels: {key_press_count} ({key_press_count/len(labels)*100:.1f}%)")
                print(f"No press labels: {no_press_count} ({no_press_count/len(labels)*100:.1f}%)")
                print(f"Total segments: {len(data.get('key_press_segments', []))}")
                
        except Exception as e:
            print(f"Error analyzing labels: {e}")
            
    def validate_labels(self, labels_file):
        """Validate label file consistency."""
        if labels_file.endswith('_merged.json'):
            labels_path = os.path.join(self.labels_dir, labels_file)
        else:
            labels_path = os.path.join(self.training_dir, labels_file)
            
        if not os.path.exists(labels_path):
            print(f"File '{labels_path}' not found")
            return False
            
        try:
            with open(labels_path, 'r') as f:
                data = json.load(f)
                
            print(f"\n=== Validation: {labels_file} ===")
            
            issues = []
            
            # Check required fields
            if 'frame_labels' in data:
                # Training data format
                required_fields = ['video_path', 'total_frames', 'fps', 'frame_labels']
                for field in required_fields:
                    if field not in data:
                        issues.append(f"Missing required field: {field}")
                        
                # Check video path
                if not os.path.exists(data['video_path']):
                    issues.append(f"Video file not found: {data['video_path']}")
                    
                # Check frame labels consistency
                frame_labels = data['frame_labels']
                expected_frames = data['total_frames']
                
                if len(frame_labels) != expected_frames:
                    issues.append(f"Frame count mismatch: expected {expected_frames}, got {len(frame_labels)}")
                    
                # Check label values
                invalid_labels = [k for k, v in frame_labels.items() if v not in [0, 1]]
                if invalid_labels:
                    issues.append(f"Invalid label values found: {len(invalid_labels)} frames")
                    
            elif 'merged_labels' in data:
                # Merged data format
                required_fields = ['videos', 'merged_labels', 'merged_segments']
                for field in required_fields:
                    if field not in data:
                        issues.append(f"Missing required field: {field}")
                        
            else:
                # Original labels format
                required_fields = ['video_path', 'total_frames', 'fps', 'labels']
                for field in required_fields:
                    if field not in data:
                        issues.append(f"Missing required field: {field}")
                        
            if issues:
                print("Issues found:")
                for issue in issues:
                    print(f"  - {issue}")
                return False
            else:
                print("✓ Labels file is valid")
                return True
                
        except Exception as e:
            print(f"Error validating labels: {e}")
            return False

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Video Label Management Tool")
    parser.add_argument('action', choices=['list', 'convert', 'merge', 'analyze', 'validate'],
                       help='Action to perform')
    parser.add_argument('--file', help='Label file name')
    parser.add_argument('--files', nargs='+', help='Multiple label files for merging')
    parser.add_argument('--output', help='Output name for merged file')
    
    args = parser.parse_args()
    
    manager = LabelManager()
    
    if args.action == 'list':
        print("=== Label Files ===")
        label_files = manager.list_label_files()
        for i, file in enumerate(label_files, 1):
            print(f"{i}. {file}")
            
        print("\n=== Training Data Files ===")
        training_files = manager.list_training_files()
        for i, file in enumerate(training_files, 1):
            print(f"{i}. {file}")
            
    elif args.action == 'convert':
        if not args.file:
            print("Please specify --file")
            return
        manager.convert_labels_to_training(args.file)
        
    elif args.action == 'merge':
        if not args.files or not args.output:
            print("Please specify --files and --output")
            return
        manager.merge_label_files(args.files, args.output)
        
    elif args.action == 'analyze':
        if not args.file:
            print("Please specify --file")
            return
        manager.analyze_labels(args.file)
        
    elif args.action == 'validate':
        if not args.file:
            print("Please specify --file")
            return
        manager.validate_labels(args.file)

if __name__ == "__main__":
    main()
