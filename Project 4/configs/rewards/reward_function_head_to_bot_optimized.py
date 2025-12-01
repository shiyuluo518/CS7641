def reward_function(params):
    '''
    OPTIMIZED Head-to-Bot Race Reward Function
    
    Key Optimizations:
    1. Proper speed normalization (0.6 m/s max from action space)
    2. Better progress tracking using waypoints
    3. Competitive positioning rewards (relative to competitors)
    4. Steering smoothness for overtaking maneuvers
    5. Balanced reward scaling (prevents reward hacking)
    6. Dynamic collision avoidance zones
    
    Improvements:
    - More accurate speed rewards
    - Better overtaking behavior
    - Smoother racing maneuvers
    - Improved competitive positioning
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
    
    # Maximum speed from action space (0.6 m/s)
    MAX_SPEED = 0.6
    
    # Try to get competitor-related parameters
    closest_competitor_distance = params.get('closest_competitor_distance', None)
    competitor_positions = params.get('competitor_positions', None)
    is_in_front = params.get('is_in_front', None)
    
    # Base penalty for going off track
    if not all_wheels_on_track:
        return 1e-3
    
    # ============================================
    # Component 1: Progress-Based Rewards (Primary - Optimized)
    # ============================================
    # Use waypoint-based progress for accuracy
    progress_reward = 0.0
    
    if closest_waypoints is not None:
        waypoint_progress = progress / 100.0
        # Strong progress reward - primary driver for competitive behavior
        progress_reward = waypoint_progress * 8.0  # Slightly reduced from 10.0 for balance
    else:
        if steps > 0:
            progress_rate = progress / max(steps, 1)
            progress_reward = progress_rate * 8.0
    
    # Completion bonus (scaled appropriately)
    completion_bonus = 0.0
    if progress >= 100:
        # Large bonus for winning/completing first
        # Scale with speed to encourage fast completion
        normalized_speed = min(speed / MAX_SPEED, 1.0)
        completion_bonus = 40.0 * normalized_speed  # Speed-scaled completion
    elif progress > 95:
        completion_bonus = 12.0
    elif progress > 90:
        completion_bonus = 6.0
    elif progress > 75:
        completion_bonus = 3.0
    elif progress > 50:
        completion_bonus = 1.5
    
    # ============================================
    # Component 2: Velocity Optimization (Corrected)
    # ============================================
    # Proper speed normalization based on actual max speed
    normalized_speed = min(speed / MAX_SPEED, 1.0)
    velocity_squared = normalized_speed ** 2
    
    # Velocity reward (slightly reduced for balance)
    velocity_reward = velocity_squared * 2.5  # Reduced from 3.0
    
    # ============================================
    # Component 3: Collision Avoidance (Optimized)
    # ============================================
    # Dynamic collision avoidance zones
    CRITICAL_DISTANCE = 0.25  # Very close - severe penalty
    WARNING_DISTANCE = 0.45   # Close - moderate penalty
    SAFE_DISTANCE = 0.6        # Safe - no penalty
    
    collision_penalty_factor = 1.0
    
    if closest_competitor_distance is not None:
        if closest_competitor_distance < CRITICAL_DISTANCE:
            # Critical: 95% reward reduction
            collision_penalty_factor = 0.05
        elif closest_competitor_distance < WARNING_DISTANCE:
            # Warning: 55% reward reduction
            collision_penalty_factor = 0.45
        elif closest_competitor_distance < SAFE_DISTANCE:
            # Caution: 15% reward reduction
            collision_penalty_factor = 0.85
        else:
            # Safe: full reward
            collision_penalty_factor = 1.0
    
    # ============================================
    # Component 4: Competitive Positioning (Enhanced)
    # ============================================
    positioning_bonus = 0.0
    
    if is_in_front is not None:
        if is_in_front:
            # Bonus for leading position
            positioning_bonus = 3.0  # Increased from 2.0
        else:
            # Small penalty for trailing (encourages overtaking)
            positioning_bonus = -0.5
    
    # Track position component (relaxed for racing line exploration)
    marker_1 = 0.18 * track_width
    marker_2 = 0.35 * track_width
    marker_3 = 0.65 * track_width
    
    if distance_from_center <= marker_1:
        track_position_reward = 0.6
    elif distance_from_center <= marker_2:
        track_position_reward = 0.4
    elif distance_from_center <= marker_3:
        track_position_reward = 0.2
    else:
        track_position_reward = 0.05
    
    # ============================================
    # Component 5: Overtaking Behaviors (Enhanced)
    # ============================================
    # Overtaking is rewarded through progress increases
    # Additional bonus for maintaining high speed while avoiding collisions
    speed_bonus = 0.0
    
    if closest_competitor_distance is None or closest_competitor_distance > SAFE_DISTANCE:
        # Safe to go fast - full speed bonus
        speed_bonus = velocity_squared * 2.0
    elif closest_competitor_distance > WARNING_DISTANCE:
        # Moderate speed bonus when competitors are at safe distance
        speed_bonus = velocity_squared * 1.0
    else:
        # Reduced speed bonus when close to competitors
        speed_bonus = velocity_squared * 0.3
    
    # ============================================
    # Component 6: Steering Smoothness (NEW)
    # ============================================
    # Smooth steering is important for competitive racing
    # Prevents oscillatory behavior that slows down lap times
    steering_penalty = 0.0
    if abs(steering_angle) > 25:  # Very large steering
        steering_penalty = -0.2 * (abs(steering_angle) / 30.0)
    elif abs(steering_angle) > 15:  # Large steering
        steering_penalty = -0.1 * (abs(steering_angle) / 20.0)
    
    # ============================================
    # Component 7: Efficiency Metric
    # ============================================
    # Progress rate weighted by velocity squared
    if steps > 0:
        progress_rate = progress / max(steps, 1)
        efficiency_reward = progress_rate * velocity_squared * 3.0
    else:
        efficiency_reward = 0.0
    
    # ============================================
    # Combine Components
    # ============================================
    base_reward = (
        progress_reward +           # Primary: progress-based
        velocity_reward +            # Speed optimization
        track_position_reward +     # Track adherence
        positioning_bonus +          # Competitive positioning
        speed_bonus +                # Conditional speed bonus
        efficiency_reward            # Efficiency metric
    )
    
    # Apply collision penalty to base reward (except completion bonus)
    reward = (base_reward - completion_bonus) * collision_penalty_factor + completion_bonus
    
    # Apply steering penalty
    reward += steering_penalty
    
    # Penalty for very slow speeds (when safe from competitors)
    if (closest_competitor_distance is None or closest_competitor_distance > SAFE_DISTANCE) and speed < 0.2:
        reward *= 0.5
    
    # Ensure positive reward
    reward = max(reward, 1e-3)
    
    return float(reward)

