import cv2
import numpy as np
import time
from collections import deque

class FingerMotionAnalyzer:
    
    def __init__(self, history_length=10, voting_window=5):
        """
        Initialize the finger motion analyzer with velocity and acceleration tracking
        
        Args:
            history_length: Number of frames to keep in history for calculations
            voting_window: Number of frames to consider for the voting mechanism
        """
        # Configuration parameters
        self.history_length = history_length  # Keep longer history for motion analysis
        self.voting_window = voting_window    # Window for voting (should be smaller than history)
        
        # Motion thresholds
        self.velocity_threshold = 4.0        # Minimum velocity for press detection (pixels/frame)
        self.acceleration_threshold = 0.8    # Minimum acceleration for press detection
        self.deceleration_threshold = -1.0   # Deceleration threshold for detecting the end of press
        self.horizontal_threshold = 15       # Maximum horizontal movement allowed (pixels)
        
        # Filtering parameters to reduce noise
        self.max_velocity_change = 20.0      # Maximum allowed velocity change between frames
        self.max_acceleration_change = 15.0  # Maximum allowed acceleration change between frames
        self.position_smoothing_factor = 0.3 # Factor for position smoothing (0-1, lower = more smoothing)
        
        # Press detection parameters
        self.press_cooldown = 1           # Seconds to wait before detecting another press
        self.press_flash_duration = 1      # Show press indicator for this duration
        self.last_press_time = 0             # Time of last detected press
        self.press_detected_time = None      # When the current press was detected
        
        # Data structures for tracking
        self.position_history = {}           # Hand_id -> Finger_id -> Deque of positions
        self.velocity_history = {}           # Hand_id -> Finger_id -> Deque of velocities 
        self.acceleration_history = {}       # Hand_id -> Finger_id -> Deque of accelerations
        self.vote_history = {}               # Hand_id -> Deque of voted finger indices
        self.last_positions = {}             # Hand_id -> Finger_id -> Last stable position for smoothing
        
        # Current state
        self.is_pressing = False
        self.finger_pressing = None          # (hand_id, finger_id) of pressing finger
        self.debug_info = {}                 # Debug information to display
          # Finger names for display
        self.finger_names = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
        self.fingertip_indices = [4, 8, 12, 16, 20]  # MediaPipe indices for fingertips

    def _initialize_tracking(self, hand_id, finger_id):
        """Initialize tracking data structures for a new hand/finger combination"""
        # Initialize position history if needed
        if hand_id not in self.position_history:
            self.position_history[hand_id] = {}
            self.velocity_history[hand_id] = {}
            self.acceleration_history[hand_id] = {}
            self.vote_history[hand_id] = deque(maxlen=self.voting_window)
            
        if finger_id not in self.position_history[hand_id]:
            self.position_history[hand_id][finger_id] = deque(maxlen=self.history_length)
            self.velocity_history[hand_id][finger_id] = deque(maxlen=self.history_length-1)
            self.acceleration_history[hand_id][finger_id] = deque(maxlen=self.history_length-2)
            
        # Initialize the finger's entry in last_positions if it doesn't exist
        key = (hand_id, finger_id)
        if key not in self.last_positions:
            self.last_positions[key] = None
    
    def update_finger_position(self, hand_id, finger_id, position, timestamp):
        """
        Update position history and calculate velocity and acceleration
        
        Args:
            hand_id: Identifier for the hand (usually 0 or 1)
            finger_id: Identifier for the finger (0-4, thumb to pinky)
            position: (x, y) tuple of finger position
            timestamp: Current timestamp
        """
        self._initialize_tracking(hand_id, finger_id)
        
        # Apply position smoothing to reduce noise
        smoothed_position = self._smooth_position(hand_id, finger_id, position)
        
        # Add new position to history
        self.position_history[hand_id][finger_id].append((smoothed_position, timestamp))
        
        # Calculate velocity if we have at least 2 positions
        if len(self.position_history[hand_id][finger_id]) >= 2:
            (prev_pos, prev_time) = self.position_history[hand_id][finger_id][-2]
            (curr_pos, curr_time) = self.position_history[hand_id][finger_id][-1]
            
            # Time difference in seconds
            time_diff = curr_time - prev_time
            if time_diff > 0:
                # Calculate velocity vector (x_vel, y_vel) in pixels/second
                x_vel = (curr_pos[0] - prev_pos[0]) / time_diff
                y_vel = (curr_pos[1] - prev_pos[1]) / time_diff
                
                # Apply velocity filtering to prevent sudden spikes
                x_vel, y_vel = self._filter_velocity(hand_id, finger_id, x_vel, y_vel)
                
                # Store velocity
                self.velocity_history[hand_id][finger_id].append((x_vel, y_vel, curr_time))
        
        # Calculate acceleration if we have at least 2 velocities
        if len(self.velocity_history[hand_id][finger_id]) >= 2:
            (prev_x_vel, prev_y_vel, prev_time) = self.velocity_history[hand_id][finger_id][-2]
            (curr_x_vel, curr_y_vel, curr_time) = self.velocity_history[hand_id][finger_id][-1]
            
            # Time difference in seconds
            time_diff = curr_time - prev_time
            if time_diff > 0:
                # Calculate acceleration vector (x_acc, y_acc) in pixels/second²
                x_acc = (curr_x_vel - prev_x_vel) / time_diff
                y_acc = (curr_y_vel - prev_y_vel) / time_diff
                
                # Apply acceleration filtering
                x_acc, y_acc = self._filter_acceleration(hand_id, finger_id, x_acc, y_acc)
                
                # Store acceleration
                self.acceleration_history[hand_id][finger_id].append((x_acc, y_acc, curr_time))
    def _smooth_position(self, hand_id, finger_id, position):
        """
        Apply smoothing to finger position to reduce jitter
        
        Args:
            hand_id: Hand identifier
            finger_id: Finger identifier
            position: Current raw position (x, y)
            
        Returns:
            Smoothed position (x, y)
        """
        key = (hand_id, finger_id)
        
        # If this is the first position for this finger, just use it directly
        if key not in self.last_positions or self.last_positions[key] is None:
            self.last_positions[key] = position
            return position
            
        # Get the last stable position
        last_pos = self.last_positions[key]
        
        # Apply exponential smoothing to reduce jitter
        alpha = self.position_smoothing_factor
        smoothed_x = alpha * position[0] + (1 - alpha) * last_pos[0]
        smoothed_y = alpha * position[1] + (1 - alpha) * last_pos[1]
        
        # Update last position
        self.last_positions[key] = (smoothed_x, smoothed_y)
        
        return (smoothed_x, smoothed_y)
    
    def _filter_velocity(self, hand_id, finger_id, x_vel, y_vel):
        """
        Filter velocity to prevent sudden spikes
        
        Args:
            hand_id: Hand identifier
            finger_id: Finger identifier
            x_vel: X velocity component
            y_vel: Y velocity component
            
        Returns:
            Filtered velocity (x_vel, y_vel)
        """
        # If we don't have previous velocities, just return the current one
        if len(self.velocity_history[hand_id][finger_id]) == 0:
            return x_vel, y_vel
            
        # Get the last velocity
        last_x_vel, last_y_vel, _ = self.velocity_history[hand_id][finger_id][-1]
        
        # Calculate change in velocity
        x_vel_change = abs(x_vel - last_x_vel)
        y_vel_change = abs(y_vel - last_y_vel)
        
        # If change is too large, clamp it
        if x_vel_change > self.max_velocity_change:
            # Preserve direction but limit magnitude of change
            x_vel = last_x_vel + (self.max_velocity_change if x_vel > last_x_vel else -self.max_velocity_change)
            
        if y_vel_change > self.max_velocity_change:
            # Preserve direction but limit magnitude of change
            y_vel = last_y_vel + (self.max_velocity_change if y_vel > last_y_vel else -self.max_velocity_change)
            
        return x_vel, y_vel
    
    def _filter_acceleration(self, hand_id, finger_id, x_acc, y_acc):
        """
        Filter acceleration to prevent sudden spikes
        
        Args:
            hand_id: Hand identifier
            finger_id: Finger identifier
            x_acc: X acceleration component
            y_acc: Y acceleration component
            
        Returns:
            Filtered acceleration (x_acc, y_acc)
        """
        # If we don't have previous accelerations, just return the current one
        if len(self.acceleration_history[hand_id][finger_id]) == 0:
            return x_acc, y_acc
            
        # Get the last acceleration
        last_x_acc, last_y_acc, _ = self.acceleration_history[hand_id][finger_id][-1]
        
        # Calculate change in acceleration
        x_acc_change = abs(x_acc - last_x_acc)
        y_acc_change = abs(y_acc - last_y_acc)
        
        # If change is too large, clamp it
        if x_acc_change > self.max_acceleration_change:
            # Preserve direction but limit magnitude of change
            x_acc = last_x_acc + (self.max_acceleration_change if x_acc > last_x_acc else -self.max_acceleration_change)
            
        if y_acc_change > self.max_acceleration_change:
            # Preserve direction but limit magnitude of change
            y_acc = last_y_acc + (self.max_acceleration_change if y_acc > last_y_acc else -self.max_acceleration_change)
            
        return x_acc, y_acc
    
    def analyze_motion(self, hand_landmarks, frame, keyboard_bbox):
        """
        Analyze finger motion to detect key presses
        
        Args:
            hand_landmarks: List of hand landmark data from MediaPipe
            frame: Current video frame
            keyboard_bbox: Bounding box of keyboard (x1, y1, x2, y2)
            
        Returns:
            is_pressing: Whether a key press is detected
            pressing_finger: (hand_id, finger_id) of the pressing finger
        """
        # Reset debug info
        self.debug_info = {}
        
        # Reset pressing state if flash duration has passed
        if self.press_detected_time and time.time() - self.press_detected_time > self.press_flash_duration:
            self.is_pressing = False
            self.finger_pressing = None
        
        # Return early if we don't have hand landmarks or keyboard bbox
        if not hand_landmarks or len(hand_landmarks) == 0 or not keyboard_bbox:
            return False, None
        
        # Extract keyboard coordinates
        kx1, ky1, kx2, ky2 = keyboard_bbox
        keyboard_height = ky2 - ky1
        
        # Current timestamp
        current_time = time.time()
        
        # Process each hand
        finger_scores = {}  # (hand_id, finger_id) -> score
        
        for hand_id, landmarks in enumerate(hand_landmarks):
            h, w, _ = frame.shape
            
            # Track each fingertip
            for finger_id, tip_idx in enumerate(self.fingertip_indices):
                # Convert landmark to pixel coordinates
                x = int(landmarks.landmark[tip_idx].x * w)
                y = int(landmarks.landmark[tip_idx].y * h)
                
                # Update motion history
                self.update_finger_position(hand_id, finger_id, (x, y), current_time)
                
                # Skip if finger is not over keyboard
                if not (kx1-10 <= x <= kx2+10 and ky1-10 <= y <= ky2+10):
                    continue
                
                # Skip if we don't have enough history
                if len(self.velocity_history[hand_id][finger_id]) < 2 or len(self.acceleration_history[hand_id][finger_id]) < 1:
                    continue
                
                # Get current velocity and acceleration
                (x_vel, y_vel, _) = self.velocity_history[hand_id][finger_id][-1]
                (x_acc, y_acc, _) = self.acceleration_history[hand_id][finger_id][-1]
                
                # Calculate score based on vertical velocity and acceleration
                # Higher score for downward movement (positive y velocity) with high acceleration
                score = 0
                
                # Vertical velocity component (downward is positive)
                if y_vel > self.velocity_threshold:
                    score += y_vel * 0.5
                
                # Vertical acceleration component
                if y_acc > self.acceleration_threshold:
                    score += y_acc * 0.3
                
                # Horizontal component penalty (we want mostly vertical movement)
                horizontal_penalty = abs(x_vel) / 10 if abs(x_vel) > self.horizontal_threshold else 0
                score -= horizontal_penalty
                
                # Depth into keyboard bonus (deeper press is more likely)
                depth_ratio = (y - ky1) / keyboard_height
                if depth_ratio > 0.25:  # At least 25% into keyboard
                    score += depth_ratio * 5
                else:
                    score = 0  # Not deep enough
                
                # Store finger score if positive
                if score > 0:
                    finger_scores[(hand_id, finger_id)] = score
                
                # Store debug info
                self.debug_info[(hand_id, finger_id)] = {
                    "position": (x, y),
                    "velocity": (x_vel, y_vel),
                    "acceleration": (x_acc, y_acc),
                    "score": score,
                    "depth_ratio": depth_ratio
                }
        
        # Find finger with highest score
        best_finger = None
        best_score = 0
        
        for finger, score in finger_scores.items():
            if score > best_score:
                best_score = score
                best_finger = finger
        
        # If we have a candidate, add it to voting history
        if best_finger and best_score > 10:  # Minimum threshold to be considered
            hand_id = best_finger[0]
            self.vote_history[hand_id].append(best_finger[1])
            
            # Only detect press if cooldown period has passed
            if current_time - self.last_press_time > self.press_cooldown:
                # Determine winner by voting (most frequent finger in history)
                if len(self.vote_history[hand_id]) >= 3:  # Need at least 3 votes
                    finger_counts = {}
                    for finger_id in self.vote_history[hand_id]:
                        if finger_id not in finger_counts:
                            finger_counts[finger_id] = 0
                        finger_counts[finger_id] += 1
                    
                    # Find finger with most votes
                    winner_finger = max(finger_counts, key=finger_counts.get)
                    winner_count = finger_counts[winner_finger]
                    
                    # If winner has enough votes (majority)
                    if winner_count >= len(self.vote_history[hand_id]) * 0.6:
                        self.is_pressing = True
                        self.finger_pressing = (hand_id, winner_finger)
                        self.press_detected_time = current_time
                        self.last_press_time = current_time
        
        return self.is_pressing, self.finger_pressing
    
    def draw_motion_analysis(self, frame):
        """Draw motion analysis visualization on the frame"""
        # Draw debug info for tracked fingers
        for (hand_id, finger_id), info in self.debug_info.items():
            x, y = info["position"]
            
            # Choose color based on score
            score = info.get("score", 0)
            if score > 10:
                color = (0, 255, 0)  # Green for high score
            elif score > 0:
                color = (0, 255, 255)  # Yellow for medium score
            else:
                color = (128, 128, 128)  # Gray for low score
            
            # Draw circle at fingertip
            cv2.circle(frame, (x, y), 5, color, -1)
            
            # Draw velocity vector
            if "velocity" in info:
                vx, vy = info["velocity"]
                # Scale down velocity for visualization
                end_x = int(x + vx * 0.05)
                end_y = int(y + vy * 0.05)
                cv2.line(frame, (x, y), (end_x, end_y), color, 2)
            
            # Draw text with finger info
            text = f"{self.finger_names[finger_id][0]}"
            if "score" in info:
                text += f": {info['score']:.1f}"
            cv2.putText(frame, text, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw press notification if detected
        if self.is_pressing and self.finger_pressing:
            hand_id, finger_id = self.finger_pressing
            
            # Calculate intensity based on time since detection (fade out effect)
            time_since_press = time.time() - self.press_detected_time
            if time_since_press < self.press_flash_duration:
                # Fade out effect
                alpha = 1.0 - (time_since_press / self.press_flash_duration)
                color_intensity = int(255 * alpha)
                color = (0, 0, color_intensity)  # Fading red
                
                # Position notification at top of screen
                finger_name = self.finger_names[finger_id]
                hand_name = "LEFT" if hand_id == 0 else "RIGHT"
                
                cv2.putText(frame, f"KEY PRESS: {hand_name} {finger_name}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Draw red circle on the pressing finger if in debug info
                if (hand_id, finger_id) in self.debug_info:
                    x, y = self.debug_info[(hand_id, finger_id)]["position"]
                    cv2.circle(frame, (x, y), 10, color, -1)
        
        return frame

    def get_key_at_position(self, keyboard_grid):
        """
        Get the key at the position of the pressing finger
        
        Args:
            keyboard_grid: KeyboardGrid object with get_key_at_position method
            
        Returns:
            key: The pressed key, or None if no key is pressed
        """
        if self.is_pressing and self.finger_pressing:
            hand_id, finger_id = self.finger_pressing
            
            # If we have debug info for this finger, get the position
            if (hand_id, finger_id) in self.debug_info:
                x, y = self.debug_info[(hand_id, finger_id)]["position"]
                return keyboard_grid.get_key_at_position(x, y)
        
        return None