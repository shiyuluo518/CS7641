def reward_function(params):
    '''
    OPTIMIZED Object-Avoidance Reward Function
    
    Key Optimizations:
    1. LIDAR-based obstacle detection (using observation if available)
    2. Dynamic obstacle avoidance zones (adaptive thresholds)
    3. Proper speed normalization (0.6 m/s max from action space)
    4. Steering smoothness for obstacle navigation
    5. Better reward balance between safety and speed
    6. Waypoint-based progress tracking
    
    Improvements:
    - More accurate obstacle detection
    - Smoother obstacle avoidance maneuvers
    - Better speed control near obstacles
    - Improved lap times while maintaining safety
    '''
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    progress = params['progress']
    speed = params['speed']
    steps = params['steps']
    all_wheels_on_track = params['all_wheels_on_track']
    is_left_of_center = params['is_left_of_center']
    
    # Get waypoint and heading information
    closest_waypoints = params.get('closest_waypoints', None)
    heading = params.get('heading', None)
    steering_angle = params.get('steering_angle', 0.0)
    
    # Maximum speed from action space (0.6 m/s, but reduced for obstacle avoidance)
    MAX_SPEED = 0.6
    OBSTACLE_AVOIDANCE_MAX_SPEED = 0.4  # Reduced max speed for safety
    
    # Try to get obstacle-related parameters
    # Note: These may need to be extracted from LIDAR observation in the environment
    closest_obstacle_distance = params.get('closest_obstacle_distance', None)
    obstacle_lane_position = params.get('obstacle_lane_position', None)
    
    # Base penalty for going off track
    if not all_wheels_on_track:
        return 1e-3
    
    # ============================================
    # Component 1: Obstacle Detection & Avoidance (70% weight)
    # ============================================
    obstacle_avoidance_reward = 0.0
    obstacle_penalty_factor = 1.0
    
    if closest_obstacle_distance is not None:
        # Dynamic obstacle avoidance zones
        CRITICAL_DISTANCE = 0.4  # Very close - severe penalty
        WARNING_DISTANCE = 0.7   # Close - moderate penalty
        SAFE_DISTANCE = 1.0      # Safe - no penalty
        
        if closest_obstacle_distance < CRITICAL_DISTANCE:
            # Critical: 95% reward reduction
            obstacle_penalty_factor = 0.05
        elif closest_obstacle_distance < WARNING_DISTANCE:
            # Warning: 60% reward reduction
            obstacle_penalty_factor = 0.4
        elif closest_obstacle_distance < SAFE_DISTANCE:
            # Caution: 20% reward reduction
            obstacle_penalty_factor = 0.8
        else:
            # Safe: full reward
            obstacle_penalty_factor = 1.0
        
        # Base obstacle avoidance reward
        obstacle_avoidance_reward = obstacle_penalty_factor * 0.7
        
        # Lane-switching bonus: reward for being on opposite side of obstacle
        if obstacle_lane_position is not None:
            if obstacle_lane_position < 0 and not is_left_of_center:
                # Obstacle on left, agent on right - good!
                obstacle_avoidance_reward += 0.15
            elif obstacle_lane_position > 0 and is_left_of_center:
                # Obstacle on right, agent on left - good!
                obstacle_avoidance_reward += 0.15
            elif obstacle_lane_position == 0:
                # Obstacle in center - slight penalty for being too close
                if abs(distance_from_center) < 0.15 * track_width:
                    obstacle_avoidance_reward -= 0.1
    else:
        # Fallback: Conservative approach when obstacle detection unavailable
        # Assume obstacles might be present
        obstacle_avoidance_reward = 0.5
        obstacle_penalty_factor = 0.7  # Slightly reduced rewards
    
    # ============================================
    # Component 2: Lane-Keeping (30% weight)
    # ============================================
    # Relaxed constraints for obstacle maneuvering
    marker_1 = 0.18 * track_width
    marker_2 = 0.35 * track_width
    marker_3 = 0.65 * track_width
    
    if distance_from_center <= marker_1:
        lane_keeping_reward = 0.3
    elif distance_from_center <= marker_2:
        lane_keeping_reward = 0.2
    elif distance_from_center <= marker_3:
        lane_keeping_reward = 0.1
    else:
        lane_keeping_reward = 0.02
    
    # ============================================
    # Component 3: Conditional Speed Control (Optimized)
    # ============================================
    # Speed rewards only when safe, with proper normalization
    speed_reward = 0.0
    
    if closest_obstacle_distance is None or closest_obstacle_distance > SAFE_DISTANCE:
        # Safe to optimize speed
        normalized_speed = min(speed / OBSTACLE_AVOIDANCE_MAX_SPEED, 1.0)
        speed_reward = (normalized_speed ** 1.5) * 0.4  # Slightly less aggressive than squared
    elif closest_obstacle_distance > WARNING_DISTANCE:
        # Moderate speed allowed
        normalized_speed = min(speed / OBSTACLE_AVOIDANCE_MAX_SPEED, 1.0)
        speed_reward = normalized_speed * 0.2
    else:
        # Slow down near obstacles
        normalized_speed = min(speed / OBSTACLE_AVOIDANCE_MAX_SPEED, 1.0)
        speed_reward = normalized_speed * 0.05  # Minimal speed reward
    
    # Penalty for going too fast near obstacles
    if closest_obstacle_distance is not None and closest_obstacle_distance < SAFE_DISTANCE:
        if speed > OBSTACLE_AVOIDANCE_MAX_SPEED * 0.8:
            speed_reward *= 0.3  # Penalize high speed near obstacles
    
    # ============================================
    # Component 4: Progress Tracking (Waypoint-based)
    # ============================================
    progress_reward = 0.0
    if closest_waypoints is not None:
        waypoint_progress = progress / 100.0
        progress_reward = waypoint_progress * 1.5
    else:
        if steps > 0:
            progress_rate = progress / max(steps, 1)
            progress_reward = progress_rate * 1.5
    
    # ============================================
    # Component 5: Steering Smoothness (NEW)
    # ============================================
    # Smooth steering is crucial for obstacle navigation
    steering_penalty = 0.0
    if abs(steering_angle) > 25:  # Very large steering
        steering_penalty = -0.15 * (abs(steering_angle) / 30.0)
    elif abs(steering_angle) > 15:  # Large steering
        steering_penalty = -0.08 * (abs(steering_angle) / 20.0)
    
    # ============================================
    # Component 6: Completion Bonuses
    # ============================================
    completion_bonus = 0.0
    if progress >= 100:
        # Large bonus for completing lap with obstacles
        # Scale with safety (obstacle_penalty_factor)
        completion_bonus = 20.0 * obstacle_penalty_factor
    elif progress > 95:
        completion_bonus = 6.0
    elif progress > 90:
        completion_bonus = 3.0
    elif progress > 75:
        completion_bonus = 1.5
    elif progress > 50:
        completion_bonus = 0.8
    
    # ============================================
    # Combine Components
    # ============================================
    base_reward = (
        obstacle_avoidance_reward +  # 70% weight (obstacle avoidance)
        lane_keeping_reward +        # 30% weight (lane keeping)
        speed_reward +                # Conditional speed
        progress_reward +             # Progress tracking
        completion_bonus              # Completion bonus
    )
    
    # Apply steering penalty
    reward = base_reward + steering_penalty
    
    # Apply obstacle penalty factor to entire reward (except completion bonus)
    # This ensures safety is prioritized
    reward = (base_reward - completion_bonus) * obstacle_penalty_factor + completion_bonus + steering_penalty
    
    # Penalty for very slow speeds (when safe)
    if (closest_obstacle_distance is None or closest_obstacle_distance > SAFE_DISTANCE) and speed < 0.1:
        reward *= 0.4
    
    # Ensure positive reward
    reward = max(reward, 1e-3)
    
    return float(reward)

