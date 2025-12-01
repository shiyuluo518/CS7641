def reward_function(params):
    '''
    Object-Avoidance Reward Function (Phase 4)
    
    This reward function extends the time-trial reward function to incorporate
    obstacle avoidance objectives. The architecture employs weighted reward composition:
    
    1. Lane-Keeping Component (30% weight): Maintains track adherence
    2. Obstacle Avoidance Component (70% weight): Distance-based penalty scaling
    3. Lane-Switching Logic: Rewards occupying lane opposite to obstacles
    4. Conditional Speed Bonuses: Only when obstacle proximity exceeds safe thresholds
    
    Key features:
    - Proximity to obstacles within 0.5m triggers severe reward reduction (90%)
    - Distances 0.5-1.0m incur moderate penalty (50%)
    - Distances beyond 1.0m permit speed optimization
    - Lane-switching logic encourages dynamic path planning
    '''
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    progress = params['progress']
    speed = params['speed']
    steps = params['steps']
    all_wheels_on_track = params['all_wheels_on_track']
    is_left_of_center = params['is_left_of_center']
    
    # Try to get obstacle-related parameters (may not be available in all environments)
    # These would typically come from LIDAR processing or environment-specific parameters
    closest_obstacle_distance = params.get('closest_obstacle_distance', None)
    obstacle_lane_position = params.get('obstacle_lane_position', None)  # -1 for left, 0 for center, 1 for right
    
    # Base penalty for going off track
    if not all_wheels_on_track:
        return 1e-3
    
    # ============================================
    # Component 1: Lane-Keeping (30% weight)
    # ============================================
    # Similar to time-trial but with relaxed constraints for maneuvering
    marker_1 = 0.2 * track_width  # Wider tolerance for obstacle avoidance
    marker_2 = 0.4 * track_width
    marker_3 = 0.7 * track_width
    
    if distance_from_center <= marker_1:
        lane_keeping_reward = 0.3  # 30% of total reward
    elif distance_from_center <= marker_2:
        lane_keeping_reward = 0.2
    elif distance_from_center <= marker_3:
        lane_keeping_reward = 0.1
    else:
        lane_keeping_reward = 0.01
    
    # ============================================
    # Component 2: Obstacle Avoidance (70% weight)
    # ============================================
    obstacle_avoidance_reward = 0.0
    
    if closest_obstacle_distance is not None:
        # Distance-based penalty scaling
        if closest_obstacle_distance < 0.5:
            # Severe penalty: 90% reward reduction
            obstacle_penalty_factor = 0.1
        elif closest_obstacle_distance < 1.0:
            # Moderate penalty: 50% reward reduction
            obstacle_penalty_factor = 0.5
        else:
            # No penalty: full reward allowed
            obstacle_penalty_factor = 1.0
        
        # Base obstacle avoidance reward scales with distance
        # Closer obstacles = lower reward, but not zero (allows learning)
        obstacle_avoidance_reward = obstacle_penalty_factor * 0.7  # 70% of total reward
        
        # ============================================
        # Component 3: Lane-Switching Logic
        # ============================================
        # Reward for occupying lane opposite to obstacle position
        if obstacle_lane_position is not None:
            # If obstacle is on left, reward right-side position
            if obstacle_lane_position < 0 and not is_left_of_center:
                lane_switching_bonus = 0.2
            # If obstacle is on right, reward left-side position
            elif obstacle_lane_position > 0 and is_left_of_center:
                lane_switching_bonus = 0.2
            # If obstacle is center, slight penalty for being too close
            elif obstacle_lane_position == 0 and abs(distance_from_center) < 0.1 * track_width:
                lane_switching_bonus = -0.1
            else:
                lane_switching_bonus = 0.0
        else:
            lane_switching_bonus = 0.0
        
        obstacle_avoidance_reward += lane_switching_bonus
    else:
        # Fallback: If obstacle distance not available, use conservative approach
        # Assume obstacles might be present and reduce speed rewards
        obstacle_avoidance_reward = 0.5  # Reduced base reward
    
    # ============================================
    # Component 4: Conditional Speed Bonuses
    # ============================================
    speed_reward = 0.0
    if closest_obstacle_distance is None or closest_obstacle_distance > 1.0:
        # Only apply speed bonuses when obstacle proximity is safe (>1.0m)
        normalized_speed = min(speed / 2.0, 1.0)  # Max speed reduced to 2.0 m/s for obstacle avoidance
        speed_reward = normalized_speed * 0.3  # Speed component
    elif closest_obstacle_distance > 0.5:
        # Moderate speed allowed when obstacles are at safe distance
        normalized_speed = min(speed / 2.0, 1.0) * 0.5
        speed_reward = normalized_speed * 0.15
    else:
        # Very low speed reward when close to obstacles
        speed_reward = 0.01
    
    # ============================================
    # Progress Component
    # ============================================
    progress_rate = progress / max(steps, 1)
    progress_reward = progress_rate * 0.2  # Progress component
    
    # Completion bonus (encourages finishing laps)
    completion_bonus = 0.0
    if progress >= 100:
        completion_bonus = 15.0  # Large bonus for completing lap with obstacles
    elif progress > 90:
        completion_bonus = 3.0
    elif progress > 75:
        completion_bonus = 1.5
    elif progress > 50:
        completion_bonus = 0.8
    
    # ============================================
    # Combine Components
    # ============================================
    reward = (
        lane_keeping_reward +           # 30% weight
        obstacle_avoidance_reward +     # 70% weight (when obstacles detected)
        speed_reward +                  # Conditional speed bonus
        progress_reward +               # Progress component
        completion_bonus               # Completion bonus
    )
    
    # Penalty for very slow speeds (encourages maintaining reasonable speed when safe)
    if speed < 0.1:
        reward *= 0.3  # Strong penalty for being too slow
    
    return float(reward)

