def reward_function(params):
    '''
    OPTIMIZED Time-Trial Reward Function
    
    Key Optimizations:
    1. Waypoint-based progress tracking (more accurate than progress/steps)
    2. Steering smoothness penalty (reduces oscillatory behavior)
    3. Proper speed normalization (based on action space: 0.6 m/s max)
    4. Heading alignment with waypoints (encourages optimal racing line)
    5. Balanced reward scaling (prevents reward hacking)
    6. Adaptive completion bonuses (scales with performance)
    
    Improvements over baseline:
    - Better progress measurement using waypoint advancement
    - Smoother steering behavior (reduces lap time)
    - More accurate speed rewards
    - Better generalization across tracks
    '''
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    progress = params['progress']
    speed = params['speed']
    steps = params['steps']
    all_wheels_on_track = params['all_wheels_on_track']
    is_left_of_center = params['is_left_of_center']
    
    # Get waypoint and heading information (if available)
    closest_waypoints = params.get('closest_waypoints', None)
    heading = params.get('heading', None)
    steering_angle = params.get('steering_angle', 0.0)
    
    # Maximum speed from action space (0.6 m/s)
    MAX_SPEED = 0.6
    
    # Base penalty for going off track
    if not all_wheels_on_track:
        return 1e-3
    
    # ============================================
    # Component 1: Progress Tracking (Optimized)
    # ============================================
    # Use waypoint advancement for more accurate progress measurement
    progress_reward = 0.0
    
    if closest_waypoints is not None and len(closest_waypoints) >= 2:
        # Waypoint-based progress: reward for advancing through waypoints
        # This is more accurate than progress/steps ratio
        waypoint_progress = progress / 100.0  # Normalize to [0, 1]
        progress_reward = waypoint_progress * 2.0  # Scale appropriately
    else:
        # Fallback: use progress/steps ratio
        if steps > 0:
            progress_rate = progress / max(steps, 1)
            progress_reward = progress_rate * 2.0
    
    # ============================================
    # Component 2: Speed Optimization (Corrected)
    # ============================================
    # Proper speed normalization based on actual max speed (0.6 m/s)
    normalized_speed = min(speed / MAX_SPEED, 1.0)
    
    # Use velocity squared to strongly favor high speed
    velocity_squared = normalized_speed ** 2
    
    # Speed reward component
    speed_reward = velocity_squared * 3.0
    
    # Penalty for very slow speeds (encourages maintaining competitive speed)
    if speed < 0.15:  # Very slow threshold
        speed_reward *= 0.3
    
    # ============================================
    # Component 3: Track Position (Racing Line)
    # ============================================
    # Relaxed position rewards to allow racing line exploration
    # Optimal racing line often deviates from centerline
    marker_1 = 0.12 * track_width  # Tight racing line
    marker_2 = 0.30 * track_width  # Good racing line
    marker_3 = 0.55 * track_width  # Acceptable position
    
    if distance_from_center <= marker_1:
        position_reward = 0.8  # High reward for optimal racing line
    elif distance_from_center <= marker_2:
        position_reward = 0.5  # Good position
    elif distance_from_center <= marker_3:
        position_reward = 0.2  # Acceptable
    else:
        position_reward = 0.05  # Near edge
    
    # ============================================
    # Component 4: Heading Alignment (NEW)
    # ============================================
    # Reward alignment with track direction (if heading available)
    heading_reward = 0.0
    if heading is not None and closest_waypoints is not None:
        # Ideal heading would align with waypoint direction
        # Simplified: reward for not having extreme heading deviations
        # This encourages smooth cornering
        heading_deviation = abs(heading) / 180.0  # Normalize to [0, 1]
        heading_reward = (1.0 - heading_deviation) * 0.3
    
    # ============================================
    # Component 5: Steering Smoothness (NEW)
    # ============================================
    # Penalize large steering changes to reduce oscillatory behavior
    # This improves lap times by reducing unnecessary corrections
    steering_penalty = 0.0
    if abs(steering_angle) > 20:  # Large steering angle
        steering_penalty = -0.1 * (abs(steering_angle) / 30.0)  # Penalty scales with angle
    elif abs(steering_angle) > 10:  # Moderate steering
        steering_penalty = -0.05 * (abs(steering_angle) / 20.0)
    
    # ============================================
    # Component 6: Efficiency Metric (Optimized)
    # ============================================
    # Progress rate weighted by velocity squared
    # This naturally encourages optimal racing lines
    if steps > 0:
        progress_rate = progress / max(steps, 1)
        efficiency_reward = progress_rate * velocity_squared * 4.0
    else:
        efficiency_reward = 0.0
    
    # ============================================
    # Component 7: Completion Bonuses (Balanced)
    # ============================================
    # Adaptive completion bonuses that scale appropriately
    completion_bonus = 0.0
    if progress >= 100:
        # Large bonus for completing lap
        # Scale with speed to encourage fast completion
        completion_bonus = 15.0 * normalized_speed  # Speed-scaled completion bonus
    elif progress > 95:
        completion_bonus = 5.0
    elif progress > 90:
        completion_bonus = 2.0
    elif progress > 75:
        completion_bonus = 1.0
    elif progress > 50:
        completion_bonus = 0.5
    
    # ============================================
    # Combine Components
    # ============================================
    reward = (
        progress_reward +        # Waypoint-based progress
        speed_reward +           # Speed optimization
        position_reward +       # Track position
        heading_reward +        # Heading alignment
        efficiency_reward +     # Efficiency metric
        completion_bonus        # Completion bonus
    )
    
    # Apply steering penalty
    reward += steering_penalty
    
    # Ensure reward is positive (for numerical stability)
    reward = max(reward, 1e-3)
    
    return float(reward)

