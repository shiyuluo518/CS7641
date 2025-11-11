"""
Environment wrapper for Overcooked AI environment.

The base environment already handles agent indexing correctly - observations
and rewards returned by env.step() are already aligned. This wrapper provides
a clean interface and handles reward shaping configuration.
"""

import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import src.config as config


class OvercookedWrapper:
    """
    Wrapper for Overcooked environment.
    
    The base environment already handles agent indexing correctly - observations
    and rewards returned by env.step() are already aligned (obs[0] and rewards[0]
    correspond to the same agent). This wrapper provides a clean interface and
    handles reward shaping configuration.
    """
    
    def __init__(self, layout_name, reward_shaping=None, horizon=None):
        """
        Initialize the environment wrapper.
        
        Args:
            layout_name: Name of the layout (e.g., 'cramped_room')
            reward_shaping: Dictionary of reward shaping parameters
            horizon: Optional episode horizon (default: from config)
        """
        import src.config as config
        
        self.layout_name = layout_name
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        
        # Configure reward shaping on MDP (must be done before creating environment)
        if reward_shaping is None:
            self.reward_shaping = self._default_reward_shaping()
        else:
            self.reward_shaping = reward_shaping
        
        # Apply reward shaping to MDP (convert keys to match MDP format)
        # MDP uses 'DISH_PICKUP_REWARD' and 'SOUP_PICKUP_REWARD' (with _REWARD)
        # Config uses 'DISH_PICKUP_REW' and 'SOUP_PICKUP_REW' (with _REW)
        mdp_reward_shaping = {}
        for key, value in self.reward_shaping.items():
            # Convert config keys to MDP keys
            if key == 'DISH_PICKUP_REW':
                mdp_reward_shaping['DISH_PICKUP_REWARD'] = value
            elif key == 'SOUP_PICKUP_REW':
                mdp_reward_shaping['SOUP_PICKUP_REWARD'] = value
            elif key == 'PLACEMENT_IN_POT_REW':
                mdp_reward_shaping['PLACEMENT_IN_POT_REW'] = value
            else:
                mdp_reward_shaping[key] = value
        
        # Update MDP reward shaping parameters
        if hasattr(self.mdp, 'reward_shaping_params'):
            self.mdp.reward_shaping_params.update(mdp_reward_shaping)
        
        # Use horizon from config if not provided
        if horizon is None:
            horizon = config.HORIZON
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
    

    def _default_reward_shaping(self):
        """Default reward shaping from config."""
        return config.get_default_reward_shaping()
    
    def reset(self):
        """Reset the environment and return initial observations."""
        # OvercookedEnv.reset() returns None, so we need to get observations from state
        self.base_env.reset()
        
        # Get observations from the state using lossless_state_encoding
        state = self.base_env.state
        obs_tuple = self.mdp.lossless_state_encoding(state)
        
        # Flatten observations to get vectors
        obs = [obs.flatten().astype(np.float32) for obs in obs_tuple]
        
        # Note: Agents are randomly assigned to starting positions on reset.
        # The observation returned by env.reset() already accounts for this.
        # obs[0] is the observation for the agent at the first starting position,
        # and obs[1] is for the agent at the second starting position.
        # No manual tracking or swapping is needed.
        
        return obs
    
    def step(self, actions):
        """
        Step the environment.
        
        Args:
            actions: List of action indices for each agent [action_0, action_1]
            
        Returns:
            obs: Processed observations
            rewards: List of rewards for each agent
            done: Whether episode is done
            info: Episode info dictionary
        """
        # Convert action indices to actual actions
        joint_actions = [Action.INDEX_TO_ACTION[action] for action in actions]
        
        obs, rewards, done, info = self.base_env.step(joint_actions)
        
        # OvercookedEnv.step() returns the state object, not observations
        # Check if obs is a state object (has 'players' attribute) or None
        if obs is None or (hasattr(obs, 'players') and hasattr(obs, 'objects')):
            # Get observations from the state using lossless_state_encoding
            state = obs if obs is not None else self.base_env.state
            obs_tuple = self.mdp.lossless_state_encoding(state)
            obs = [obs.flatten().astype(np.float32) for obs in obs_tuple]
        elif isinstance(obs, tuple) and len(obs) > 0 and hasattr(obs[0], 'shape') and len(obs[0].shape) > 1:
            # If obs is a tuple of arrays that need flattening
            obs = [o.flatten().astype(np.float32) if len(o.shape) > 1 else o.astype(np.float32) for o in obs]
        
        # Use shaped rewards from info dict to help agents learn faster
        # Shaped rewards provide intermediate rewards for sub-tasks (placing onions, picking dishes, etc.)
        # This is critical for achieving ≥7 soups per episode
        if isinstance(info, dict) and 'shaped_r_by_agent' in info:
            # Use shaped rewards (includes intermediate rewards for sub-tasks)
            shaped_rewards = info['shaped_r_by_agent']
            if isinstance(shaped_rewards, (list, tuple)) and len(shaped_rewards) == 2:
                rewards = [float(shaped_rewards[0]), float(shaped_rewards[1])]
            else:
                # Fallback: split total reward equally
                total_shaped = float(sum(shaped_rewards)) if hasattr(shaped_rewards, '__iter__') else float(shaped_rewards)
                rewards = [total_shaped / 2.0, total_shaped / 2.0]
        else:
            # Fallback: use sparse rewards from env.step() (only +20 when soup delivered)
            # Convert rewards to list format (per-agent rewards)
            if isinstance(rewards, (int, float)):
                # Split total reward equally between agents
                rewards = [float(rewards) / 2.0, float(rewards) / 2.0]
            elif not isinstance(rewards, list):
                # Convert to list if it's not already
                rewards = [float(rewards), float(rewards)]
        
        # Note: The environment already handles agent indexing correctly.
        # The observation returned by env.step() already accounts for random
        # starting position assignment. obs[0] and rewards[0] correspond to
        # the same agent, and obs[1] and rewards[1] correspond to the same agent.
        # No manual swapping is needed.
        
        return obs, rewards, done, info
    
    def get_global_state(self, obs=None):
        """
        Extract true global state from environment.
        
        This constructs a compact, objective representation of the global state
        including agent positions, object locations, and pot states. This is
        more informative than simple observation concatenation for the mixing network.
        
        Args:
            obs: Optional observations (if None, uses current state)
            
        Returns:
            Global state vector
        """
        try:
            state = self.base_env.state
            state_vec = []
            
            # Agent positions (2 agents * 2 coordinates = 4)
            for player in state.players:
                pos = player.position
                state_vec.extend([float(pos[0]), float(pos[1])])
            
            # Agent orientations (2 agents * 4 one-hot = 8)
            for player in state.players:
                orientation = player.orientation
                # Convert direction to one-hot (0=up, 1=down, 2=left, 3=right)
                orient_vec = [0.0, 0.0, 0.0, 0.0]
                if orientation == (0, -1):  # up
                    orient_vec[0] = 1.0
                elif orientation == (0, 1):  # down
                    orient_vec[1] = 1.0
                elif orientation == (-1, 0):  # left
                    orient_vec[2] = 1.0
                elif orientation == (1, 0):  # right
                    orient_vec[3] = 1.0
                state_vec.extend(orient_vec)
            
            # Agent objects (2 agents * 1 value = 2)
            # 0=nothing, 1=onion, 2=tomato, 3=pot, 4=dish, 5=soup
            for player in state.players:
                obj = player.held_object
                if obj is None:
                    state_vec.append(0.0)
                else:
                    obj_name = obj.name if hasattr(obj, 'name') else str(type(obj).__name__).lower()
                    if 'onion' in obj_name:
                        state_vec.append(1.0)
                    elif 'tomato' in obj_name:
                        state_vec.append(2.0)
                    elif 'pot' in obj_name:
                        state_vec.append(3.0)
                    elif 'dish' in obj_name:
                        state_vec.append(4.0)
                    elif 'soup' in obj_name:
                        state_vec.append(5.0)
                    else:
                        state_vec.append(0.0)
            
            # Pot states (up to 2 pots * state info)
            # Try to get pots from the MDP layout
            pots = []
            if hasattr(self.mdp, 'get_pot_locations'):
                pot_locations = self.mdp.get_pot_locations()
                for pot_loc in pot_locations[:2]:  # Max 2 pots
                    # Check if there's an object at this location
                    if pot_loc in state.objects:
                        obj_list = state.objects[pot_loc]
                        for obj in obj_list:
                            if hasattr(obj, 'is_empty') or hasattr(obj, 'is_cooking'):
                                pots.append((pot_loc, obj))
                                break
            
            # Extract pot information
            for idx in range(2):  # Max 2 pots
                if idx < len(pots):
                    pot_loc, pot = pots[idx]
                    state_vec.extend([float(pot_loc[0]), float(pot_loc[1])])
                    
                    # Pot state: is_empty, is_full, is_cooking, is_ready
                    is_empty = pot.is_empty() if hasattr(pot, 'is_empty') else True
                    is_cooking = pot.is_cooking() if hasattr(pot, 'is_cooking') else False
                    is_ready = pot.is_ready() if hasattr(pot, 'is_ready') else False
                    is_full = not is_empty and not is_cooking and not is_ready
                    
                    state_vec.extend([
                        1.0 if is_empty else 0.0,
                        1.0 if is_full else 0.0,
                        1.0 if is_cooking else 0.0,
                        1.0 if is_ready else 0.0
                    ])
                    
                    # Ingredients in pot
                    if hasattr(pot, 'ingredients') and not is_empty:
                        try:
                            ingredients = pot.ingredients
                            onions = sum(1 for ing in ingredients if 'onion' in str(type(ing).__name__).lower())
                            state_vec.append(float(onions))
                        except:
                            state_vec.append(0.0)
                    else:
                        state_vec.append(0.0)
                else:
                    # Pad with zeros if fewer than 2 pots
                    state_vec.extend([0.0] * 7)  # 2 pos + 4 state + 1 ingredient = 7
            
            # Simplified: Use layout dimensions for normalization
            # Get layout size from MDP (height, width)
            try:
                if hasattr(self.mdp, 'shape'):
                    layout_shape = self.mdp.shape
                elif hasattr(self.mdp, 'terrain_mtx'):
                    layout_shape = (len(self.mdp.terrain_mtx), len(self.mdp.terrain_mtx[0]))
                else:
                    layout_shape = (10, 10)  # Default fallback
                state_vec.extend([float(layout_shape[0]), float(layout_shape[1])])
            except:
                state_vec.extend([10.0, 10.0])  # Default fallback
            
            return np.array(state_vec, dtype=np.float32)
            
        except Exception as e:
            # Fallback to observation concatenation if global state extraction fails
            # This ensures robustness if the environment API changes
            if obs is not None:
                if isinstance(obs, list):
                    return np.concatenate(obs).astype(np.float32)
                return np.array(obs, dtype=np.float32)
            # If no obs provided and extraction failed, return zeros
            # (this shouldn't happen in practice)
            return np.zeros(192, dtype=np.float32)
    
    def get_episode_metrics(self):
        """Extract episode metrics from the environment."""
        if hasattr(self.base_env, 'episode_metrics'):
            return self.base_env.episode_metrics
        return {}
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.base_env.render(mode=mode)
    
    def close(self):
        """Close the environment."""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()

