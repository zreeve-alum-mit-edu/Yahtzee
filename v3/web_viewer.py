from flask import Flask, render_template, jsonify, request
import torch
from ppo_hold import PPOHoldPlayer
from multi_yahtzee import MultiYahtzee
import os

app = Flask(__name__)

# Global game state
game_state = {
    'game': None,
    'player': None,
    'Z': 3,  # Number of scorecards (can be 1-5)
    'current_selection': 0,  # 0 to Z*13-1
    'current_roll': 0,
    'current_step': 0,  # 0=before decision, 1=after decision
    'game_started': False,
    'cumulative_score': 0,
    'last_hold_mask': None,
    'dice_values': [],
    'hold_pattern': [],
    'action_text': '',
    'scorecards': [],  # List of Z scorecards, each with scores and available
    'upper_bonuses': [],  # Bonus status for each scorecard
    'multipliers': []  # Multiplier for each scorecard
}

# Category names
CATEGORY_NAMES = [
    'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
    '3-of-Kind', '4-of-Kind', 'Full House', 'Sm Straight', 
    'Lg Straight', 'Yahtzee', 'Chance'
]

def load_model(Z):
    """Load the trained PPO Hold model for Multi-Yahtzee with Z scorecards."""
    model_path = f'ppo_hold_multi_yahtzee_z{Z}_model.pth'
    if not os.path.exists(model_path):
        # Try default Z=3 model
        model_path = 'ppo_hold_multi_yahtzee_z3_model.pth'
        if not os.path.exists(model_path):
            return False
        
    # Load checkpoint first to get architecture hyperparameters
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Get architecture hyperparameters from checkpoint (with defaults for older models)
    hidden_dim = checkpoint.get('hidden_dim', 256)
    num_shared_layers = checkpoint.get('num_shared_layers', 3)
    num_branch_layers = checkpoint.get('num_branch_layers', 2)
    activation = checkpoint.get('activation', 'leaky_relu')
    
    # Create player with same architecture as training
    game_state['player'] = PPOHoldPlayer(
        Z=Z,
        lr=1e-4,
        hidden_dim=hidden_dim,
        num_shared_layers=num_shared_layers,
        num_branch_layers=num_branch_layers,
        activation=activation,
        use_compile=False,
        use_amp=False,
        device='cpu'
    )
    
    # Handle torch.compile state dict format (removes _orig_mod prefix)
    def fix_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]  # Remove '_orig_mod.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
    
    game_state['player'].policy_net.load_state_dict(fix_state_dict(checkpoint['policy_net']))
    game_state['player'].value_net.load_state_dict(fix_state_dict(checkpoint['value_net']))
    game_state['player'].training = False
    
    # Store Z from loaded model
    if 'Z' in checkpoint:
        game_state['Z'] = checkpoint['Z']
    
    return True

@app.route('/')
def index():
    return render_template('viewer.html')

@app.route('/api/start_game', methods=['POST'])
def start_game():
    """Start a new game."""
    data = request.json or {}
    Z = data.get('Z', 3)  # Default to Z=3
    Z = min(max(Z, 1), 5)  # Clamp between 1 and 5
    
    # Load model for this Z if different
    if game_state['player'] is None or game_state['Z'] != Z:
        game_state['Z'] = Z
        if not load_model(Z):
            return jsonify({'error': f'Could not load model for Z={Z}'}), 400
    
    # Create new game
    game_state['game'] = MultiYahtzee(num_games=1, Z=Z, device='cpu')
    game_state['current_selection'] = 0
    game_state['current_roll'] = 0
    game_state['current_step'] = 0
    game_state['cumulative_score'] = 0
    game_state['game_started'] = True
    game_state['last_hold_mask'] = None
    
    # Initialize scorecards
    game_state['scorecards'] = []
    game_state['upper_bonuses'] = []
    game_state['multipliers'] = []
    for z in range(Z):
        game_state['scorecards'].append({
            'scores': {},
            'available': list(range(13))
        })
        game_state['upper_bonuses'].append(0)
        game_state['multipliers'].append(z + 1)
    
    # Initialize first selection
    game_state['game'].round[:] = 0
    game_state['game'].roll_dice()
    game_state['game'].turn.zero_()
    game_state['game'].turn[:, 0, 0] = 1
    game_state['current_roll'] = 1
    
    # Get dice values
    dice_values = torch.argmax(game_state['game'].dice[0], dim=1) + 1
    game_state['dice_values'] = dice_values.tolist()
    game_state['hold_pattern'] = []
    game_state['action_text'] = f'Game started! {Z} scorecards, {Z*13} total selections. Click Next Step.'
    
    return jsonify(get_game_state())

@app.route('/api/next_step', methods=['POST'])
def next_step():
    """Execute the next step in the game."""
    if not game_state['game_started']:
        return jsonify({'error': 'Game not started'}), 400
    
    Z = game_state['Z']
    
    if game_state['current_step'] == 0:  # Before decision
        if game_state['current_roll'] <= 2:  # Hold decision
            # Get hold decision
            hold_mask = game_state['player'].decide_hold(game_state['game'])
            game_state['last_hold_mask'] = hold_mask[0]
            
            # Show hold pattern
            game_state['hold_pattern'] = hold_mask[0].tolist()
            game_state['action_text'] = f"Agent decides hold pattern for roll {game_state['current_roll']}"
            game_state['current_step'] = 1
            
        else:  # Category decision (roll 3)
            # Get category decision
            category = game_state['player'].decide_category(game_state['game'])
            cat_idx = category[0].item()
            
            # Decode which scorecard and category
            scorecard_idx = cat_idx // 13
            local_cat_idx = cat_idx % 13
            
            game_state['action_text'] = f"Agent chooses: Scorecard {scorecard_idx} ({scorecard_idx+1}x), {CATEGORY_NAMES[local_cat_idx]}"
            
            # Score the category
            reward = game_state['game'].score_category(category)
            points = reward[0].item()  # This includes the multiplier
            game_state['cumulative_score'] += points
            
            # Calculate base score (without multiplier)
            base_score = points / (scorecard_idx + 1)
            
            # Update scorecard
            game_state['scorecards'][scorecard_idx]['scores'][local_cat_idx] = int(base_score)
            game_state['scorecards'][scorecard_idx]['available'].remove(local_cat_idx)
            
            # Check for upper bonus
            upper_total = sum(game_state['scorecards'][scorecard_idx]['scores'].get(i, 0) for i in range(6))
            if upper_total >= 63 and game_state['upper_bonuses'][scorecard_idx] == 0:
                game_state['upper_bonuses'][scorecard_idx] = 35
                bonus_points = 35 * (scorecard_idx + 1)  # Apply multiplier to bonus
                game_state['cumulative_score'] += bonus_points
            
            # Move to next selection
            game_state['current_selection'] += 1
            if game_state['current_selection'] < Z * 13:
                game_state['game'].round[:] = game_state['current_selection']
                game_state['current_roll'] = 0
                game_state['current_step'] = 1  # Will roll on next step
            else:
                # Game over
                game_state['game_started'] = False
                game_state['action_text'] = f"Game Over! Final Score: {game_state['cumulative_score']}"
                
    else:  # After decision (execute action)
        if game_state['current_roll'] == 0:  # Need to roll first dice of selection
            game_state['game'].roll_dice()
            game_state['game'].turn.zero_()
            game_state['game'].turn[:, 0, 0] = 1
            game_state['current_roll'] = 1
            game_state['action_text'] = f"Rolling dice for selection {game_state['current_selection'] + 1}/{Z*13}"
            game_state['hold_pattern'] = []
            game_state['last_hold_mask'] = None
            
        elif game_state['current_roll'] <= 2:  # Execute hold and reroll
            if game_state['last_hold_mask'] is not None:
                game_state['game'].roll_dice(game_state['last_hold_mask'].unsqueeze(0))
                game_state['current_roll'] += 1
                game_state['game'].turn.zero_()
                game_state['game'].turn[:, 0, game_state['current_roll'] - 1] = 1
                game_state['action_text'] = f"Rerolled dice - now on roll {game_state['current_roll']}"
                game_state['hold_pattern'] = []
                
        game_state['current_step'] = 0
    
    # Update dice values
    if game_state['current_roll'] > 0:
        dice_values = torch.argmax(game_state['game'].dice[0], dim=1) + 1
        game_state['dice_values'] = dice_values.tolist()
    
    return jsonify(get_game_state())

@app.route('/api/auto_play_selection', methods=['POST'])
def auto_play_selection():
    """Automatically play through the current selection (3 rolls + score)."""
    if not game_state['game_started']:
        return jsonify({'error': 'Game not started'}), 400
    
    steps = []
    initial_selection = game_state['current_selection']
    
    # Play until we complete this selection
    while game_state['game_started'] and game_state['current_selection'] == initial_selection:
        next_step()
        steps.append(get_game_state())
    
    # Add one more step if we just started a new selection
    if game_state['game_started'] and game_state['current_roll'] == 0:
        next_step()
        steps.append(get_game_state())
        
    return jsonify({
        'steps': steps,
        'final_state': get_game_state()
    })

def get_game_state():
    """Get the current game state as a dictionary."""
    Z = game_state['Z']
    return {
        'Z': Z,
        'selection': game_state['current_selection'] + 1,
        'total_selections': Z * 13,
        'roll': game_state['current_roll'],
        'step': game_state['current_step'],
        'game_started': game_state['game_started'],
        'cumulative_score': game_state['cumulative_score'],
        'dice_values': game_state['dice_values'],
        'hold_pattern': game_state['hold_pattern'],
        'action_text': game_state['action_text'],
        'scorecards': game_state['scorecards'],
        'upper_bonuses': game_state['upper_bonuses'],
        'multipliers': game_state['multipliers'],
        'category_names': CATEGORY_NAMES
    }

# Load default model on startup
model_loaded = load_model(3)

if not model_loaded:
    print("WARNING: Could not load model file ppo_hold_multi_yahtzee_z3_model.pth")

if __name__ == '__main__':
    print("Starting Multi-Yahtzee web viewer at http://localhost:5678")
    print("Supports Z=1 to Z=5 scorecards")
    app.run(debug=False, host='0.0.0.0', port=5678)