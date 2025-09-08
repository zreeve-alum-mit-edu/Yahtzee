from flask import Flask, render_template, jsonify, request
import torch
from ppo_hold import PPOHoldPlayer
from yahtzee import Yahtzee
import os

app = Flask(__name__)

# Global game state
game_state = {
    'game': None,
    'player': None,
    'current_round': 0,
    'current_roll': 0,
    'current_step': 0,  # 0=before decision, 1=after decision
    'game_started': False,
    'cumulative_score': 0,
    'last_hold_mask': None,
    'dice_values': [],
    'hold_pattern': [],
    'action_text': '',
    'scores': {},
    'available': [],
    'upper_bonus': 0
}

# Category names
CATEGORY_NAMES = [
    'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
    '3-of-Kind', '4-of-Kind', 'Full House', 'Sm Straight', 
    'Lg Straight', 'Yahtzee', 'Chance'
]

def load_model():
    """Load the trained PPO Hold model."""
    model_path = 'ppo_hold_yahtzee_model.pth'
    if not os.path.exists(model_path):
        return False
        
    # Create player with same architecture as training
    game_state['player'] = PPOHoldPlayer(
        lr=1e-4,
        hidden_dim=128,
        num_shared_layers=3,
        num_branch_layers=2,
        use_compile=False,
        use_amp=False
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
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
    
    return True

@app.route('/')
def index():
    return render_template('viewer.html')

@app.route('/api/start_game', methods=['POST'])
def start_game():
    """Start a new game."""
    # Create new game
    game_state['game'] = Yahtzee(num_games=1)
    game_state['current_round'] = 0
    game_state['current_roll'] = 0
    game_state['current_step'] = 0
    game_state['cumulative_score'] = 0
    game_state['game_started'] = True
    game_state['last_hold_mask'] = None
    game_state['scores'] = {}
    game_state['available'] = list(range(13))
    game_state['upper_bonus'] = 0
    
    # Initialize first round
    game_state['game'].round[:] = 0
    game_state['game'].roll_dice()
    game_state['game'].turn.zero_()
    game_state['game'].turn[:, 0] = 1
    game_state['current_roll'] = 1
    
    # Get dice values
    dice_values = torch.argmax(game_state['game'].dice[0], dim=1) + 1
    game_state['dice_values'] = dice_values.tolist()
    game_state['hold_pattern'] = []
    game_state['action_text'] = 'Game started! Click Next Step to see agent decisions.'
    
    return jsonify(get_game_state())

@app.route('/api/next_step', methods=['POST'])
def next_step():
    """Execute the next step in the game."""
    if not game_state['game_started']:
        return jsonify({'error': 'Game not started'}), 400
        
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
            
            game_state['action_text'] = f"Agent chooses: {CATEGORY_NAMES[cat_idx]}"
            
            # Score the category
            reward = game_state['game'].score_category(category)
            points = reward[0].item()
            game_state['cumulative_score'] += points
            
            # Update scorecard
            game_state['scores'][cat_idx] = int(points)
            game_state['available'].remove(cat_idx)
            
            # Check for upper bonus
            upper_total = sum(game_state['scores'].get(i, 0) for i in range(6))
            if upper_total >= 63 and game_state['upper_bonus'] == 0:
                game_state['upper_bonus'] = 35
                game_state['cumulative_score'] += 35
            
            # Move to next round
            game_state['current_round'] += 1
            if game_state['current_round'] < 13:
                game_state['game'].round[:] = game_state['current_round']
                game_state['current_roll'] = 0
                game_state['current_step'] = 1  # Will roll on next step
            else:
                # Game over
                game_state['game_started'] = False
                game_state['action_text'] = f"Game Over! Final Score: {game_state['cumulative_score']}"
                
    else:  # After decision (execute action)
        if game_state['current_roll'] == 0:  # Need to roll first dice of round
            game_state['game'].roll_dice()
            game_state['game'].turn.zero_()
            game_state['game'].turn[:, 0] = 1
            game_state['current_roll'] = 1
            game_state['action_text'] = f"Rolling dice for round {game_state['current_round'] + 1}"
            game_state['hold_pattern'] = []
            game_state['last_hold_mask'] = None
            
        elif game_state['current_roll'] <= 2:  # Execute hold and reroll
            if game_state['last_hold_mask'] is not None:
                game_state['game'].roll_dice(game_state['last_hold_mask'].unsqueeze(0))
                game_state['current_roll'] += 1
                game_state['game'].turn.zero_()
                game_state['game'].turn[:, game_state['current_roll'] - 1] = 1
                game_state['action_text'] = f"Rerolled dice - now on roll {game_state['current_roll']}"
                game_state['hold_pattern'] = []
                
        game_state['current_step'] = 0
    
    # Update dice values
    if game_state['current_roll'] > 0:
        dice_values = torch.argmax(game_state['game'].dice[0], dim=1) + 1
        game_state['dice_values'] = dice_values.tolist()
    
    return jsonify(get_game_state())

@app.route('/api/auto_play_round', methods=['POST'])
def auto_play_round():
    """Automatically play through the current round."""
    if not game_state['game_started']:
        return jsonify({'error': 'Game not started'}), 400
    
    steps = []
    initial_round = game_state['current_round']
    
    while game_state['game_started'] and game_state['current_round'] == initial_round:
        next_step()
        steps.append(get_game_state())
        
    return jsonify({
        'steps': steps,
        'final_state': get_game_state()
    })

def get_game_state():
    """Get the current game state as a dictionary."""
    return {
        'round': game_state['current_round'] + 1,
        'roll': game_state['current_roll'],
        'step': game_state['current_step'],
        'game_started': game_state['game_started'],
        'cumulative_score': game_state['cumulative_score'],
        'dice_values': game_state['dice_values'],
        'hold_pattern': game_state['hold_pattern'],
        'action_text': game_state['action_text'],
        'scores': game_state['scores'],
        'available': game_state['available'],
        'upper_bonus': game_state['upper_bonus'],
        'category_names': CATEGORY_NAMES
    }

# Load model on startup
model_loaded = load_model()

if not model_loaded:
    print("WARNING: Could not load model file ppo_hold_yahtzee_model.pth")

if __name__ == '__main__':
    print("Starting web viewer at http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)