def reward_function(params):
    '''
    Head-to-Bot Race Reward Function (Phase 5)
    
    This reward function prioritizes competitive positioning over conservative driving.
    The architecture emphasizes aggressive advancement while maintaining collision avoidance:
    
    1. Progress-Based Rewards: Scale directly with track completion (encourages aggressive advancement)
    2. Velocity Multipliers: Squared speed scaling (strongly favors high-speed when safe)
    3. Collision Avoidance: Graduated penalties based on proximity to competitor vehicles
    4. Overtaking Behaviors: Implicit rewards through progress-based structure
    
    Key features:
    - Distances < 0.3m: 90% reward reduction (severe collision risk)
    - Distances 0.3-0.5m: 50% reward reduction (moderate risk)
    - Distances > 0.5m: Normal reward accumulation
    - Progress-based structure naturally rewards overtaking
    '''
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    progress = params['progress']
    speed = params['speed']
    steps = params['steps']
    all_wheels_on_track = params['all_wheels_on_track']
    is_left_of_center = params['is_left_of_center']
    
    # Try to get competitor-related parameters (may not be available in all environments)
    # These would typically come from LIDAR processing or environment-specific parameters
    closest_competitor_distance = params.get('closest_competitor_distance', None)
    competitor_positions = params.get('competitor_positions', None)  # List of competitor positions
    is_in_front = params.get('is_in_front', None)  # Whether agent is ahead of competitors
    
    # Base penalty for going off track
    if not all_wheels_on_track:
        return 1e-3
    
    # ============================================
    # Component 1: Progress-Based Rewards (Primary)
    # ============================================
    # Progress directly scales with reward - encourages aggressive advancement
    progress_rate = progress / max(steps, 1)
    
    # Strong progress reward - this is the primary driver for competitive behavior
    progress_reward = progress_rate * 10.0  # High weight for progress
    
    # Completion bonus (very high for competitive racing)
    completion_bonus = 0.0
    if progress >= 100:
        completion_bonus = 50.0  # Very large bonus for winning/completing first
    elif progress > 90:
        completion_bonus = 10.0
    elif progress > 75:
        completion_bonus = 5.0
    elif progress > 50:
        completion_bonus = 2.0
    
    # ============================================
    # Component 2: Velocity Multipliers
    # ============================================
    # Squared speed scaling strongly favors high-speed operation
    normalized_speed = min(speed / 4.0, 1.0)  # Assuming max speed around 4.0 m/s
    velocity_squared = normalized_speed ** 2
    
    # Velocity reward scales with speed squared
    velocity_reward = velocity_squared * 3.0
    
    # ============================================
    # Component 3: Collision Avoidance
    # ============================================
    collision_penalty_factor = 1.0  # Default: no penalty
    
    if closest_competitor_distance is not None:
        # Graduated penalties based on proximity
        if closest_competitor_distance < 0.3:
            # Severe collision risk: 90% reward reduction
            collision_penalty_factor = 0.1
        elif closest_competitor_distance < 0.5:
            # Moderate risk: 50% reward reduction
            collision_penalty_factor = 0.5
        else:
            # Safe distance: normal reward accumulation
            collision_penalty_factor = 1.0
    
    # ============================================
    # Component 4: Competitive Positioning
    # ============================================
    positioning_bonus = 0.0
    
    if is_in_front is not None:
        # Bonus for being ahead of competitors
        if is_in_front:
            positioning_bonus = 2.0  # Bonus for leading position
        else:
            positioning_bonus = 0.0
    
    # Track position component (relaxed for racing line exploration)
    marker_1 = 0.2 * track_width
    marker_2 = 0.4 * track_width
    marker_3 = 0.7 * track_width
    
    if distance_from_center <= marker_1:
        track_position_reward = 0.5
    elif distance_from_center <= marker_2:
        track_position_reward = 0.3
    elif distance_from_center <= marker_3:
        track_position_reward = 0.1
    else:
        track_position_reward = 0.01
    
    # ============================================
    # Component 5: Overtaking Behaviors
    # ============================================
    # Overtaking is implicitly rewarded through progress increases
    # When agent successfully passes competitors, progress increases rapidly
    # This naturally reinforces the behavior through the progress_reward component
    
    # Additional bonus for maintaining high speed while avoiding collisions
    if closest_competitor_distance is None or closest_competitor_distance > 0.5:
        # Safe to go fast
        speed_bonus = velocity_squared * 2.0
    else:
        # Close to competitors - reduce speed bonus
        speed_bonus = velocity_squared * 0.5
    
    # ============================================
    # Combine Components
    # ============================================
    base_reward = (
        progress_reward +           # Primary: progress-based
        velocity_reward +           # Speed optimization
        track_position_reward +     # Track adherence
        positioning_bonus +        # Competitive positioning
        speed_bonus                 # Conditional speed bonus
    )
    
    # Apply collision penalty
    reward = base_reward * collision_penalty_factor
    
    # Add completion bonus (not affected by collision penalty)
    reward += completion_bonus
    
    # Penalty for very slow speeds (encourages maintaining competitive speed)
    if speed < 0.2:
        reward *= 0.4  # Strong penalty for being too slow in competitive racing
    
    return float(reward)

