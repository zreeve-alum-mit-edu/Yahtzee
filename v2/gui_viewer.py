import tkinter as tk
from tkinter import ttk, messagebox
import torch
from ppo_hold import PPOHoldPlayer
from yahtzee import Yahtzee
import os

class YahtzeeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Yahtzee PPO Hold Agent Viewer")
        self.root.geometry("900x800")
        
        # Game state
        self.game = None
        self.player = None
        self.current_round = 0
        self.current_roll = 0
        self.current_step = 0  # 0=before decision, 1=after decision
        self.game_started = False
        self.cumulative_score = 0
        self.last_hold_mask = None
        
        # Category names
        self.category_names = [
            'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
            '3-of-Kind', '4-of-Kind', 'Full House', 'Sm Straight', 
            'Lg Straight', 'Yahtzee', 'Chance'
        ]
        
        # Load model
        self.load_model()
        
        # Create UI
        self.create_widgets()
        
    def load_model(self):
        """Load the trained PPO Hold model."""
        model_path = 'ppo_hold_yahtzee_model.pth'
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file {model_path} not found!")
            return
            
        # Create player with same architecture as training
        self.player = PPOHoldPlayer(
            lr=1e-4,
            hidden_dim=128,
            num_shared_layers=3,
            num_branch_layers=2,
            use_compile=False,  # Disable for GUI
            use_amp=False  # Disable for GUI
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        self.player.policy_net.load_state_dict(checkpoint['policy_net'])
        self.player.value_net.load_state_dict(checkpoint['value_net'])
        self.player.training = False  # Set to evaluation mode
        
        print("Model loaded successfully!")
        
    def create_widgets(self):
        """Create the GUI widgets."""
        # Control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.start_button = ttk.Button(control_frame, text="Start New Game", command=self.start_game)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.step_button = ttk.Button(control_frame, text="Next Step", command=self.next_step, state=tk.DISABLED)
        self.step_button.grid(row=0, column=1, padx=5)
        
        self.auto_play_button = ttk.Button(control_frame, text="Auto Play Round", command=self.auto_play_round, state=tk.DISABLED)
        self.auto_play_button.grid(row=0, column=2, padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Click 'Start New Game' to begin", font=('Arial', 12, 'bold'))
        self.status_label.grid(row=0, column=3, padx=20)
        
        # Game info
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.round_label = ttk.Label(info_frame, text="Round: -/13", font=('Arial', 11))
        self.round_label.grid(row=0, column=0, padx=10)
        
        self.roll_label = ttk.Label(info_frame, text="Roll: -/3", font=('Arial', 11))
        self.roll_label.grid(row=0, column=1, padx=10)
        
        self.score_label = ttk.Label(info_frame, text="Total Score: 0", font=('Arial', 11, 'bold'))
        self.score_label.grid(row=0, column=2, padx=10)
        
        # Dice display
        dice_frame = ttk.LabelFrame(self.root, text="Current Dice", padding="10")
        dice_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        self.dice_labels = []
        self.hold_labels = []
        for i in range(5):
            dice_label = ttk.Label(dice_frame, text="?", font=('Arial', 24, 'bold'), 
                                  relief=tk.RIDGE, width=4, anchor=tk.CENTER)
            dice_label.grid(row=0, column=i, padx=5, pady=5)
            self.dice_labels.append(dice_label)
            
            hold_label = ttk.Label(dice_frame, text="", font=('Arial', 10))
            hold_label.grid(row=1, column=i, padx=5)
            self.hold_labels.append(hold_label)
        
        # Action display
        action_frame = ttk.LabelFrame(self.root, text="Current Action", padding="10")
        action_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        self.action_label = ttk.Label(action_frame, text="Waiting to start...", font=('Arial', 11))
        self.action_label.pack()
        
        # Scorecard
        scorecard_frame = ttk.LabelFrame(self.root, text="Scorecard", padding="10")
        scorecard_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        # Create scorecard grid
        headers = ['Category', 'Score', 'Available']
        for col, header in enumerate(headers):
            ttk.Label(scorecard_frame, text=header, font=('Arial', 10, 'bold')).grid(
                row=0, column=col, padx=5, pady=2, sticky=tk.W)
        
        ttk.Separator(scorecard_frame, orient='horizontal').grid(
            row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2)
        
        self.score_entries = {}
        self.available_labels = {}
        
        # Upper section
        for i, cat in enumerate(self.category_names[:6]):
            ttk.Label(scorecard_frame, text=cat).grid(row=i+2, column=0, padx=5, pady=2, sticky=tk.W)
            score_label = ttk.Label(scorecard_frame, text="-")
            score_label.grid(row=i+2, column=1, padx=5, pady=2)
            self.score_entries[i] = score_label
            avail_label = ttk.Label(scorecard_frame, text="")
            avail_label.grid(row=i+2, column=2, padx=5, pady=2)
            self.available_labels[i] = avail_label
        
        # Upper bonus
        ttk.Separator(scorecard_frame, orient='horizontal').grid(
            row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(scorecard_frame, text="Upper Bonus", font=('Arial', 10, 'italic')).grid(
            row=9, column=0, padx=5, pady=2, sticky=tk.W)
        self.bonus_label = ttk.Label(scorecard_frame, text="-")
        self.bonus_label.grid(row=9, column=1, padx=5, pady=2)
        
        # Lower section
        ttk.Separator(scorecard_frame, orient='horizontal').grid(
            row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2)
        for i, cat in enumerate(self.category_names[6:]):
            idx = i + 6
            ttk.Label(scorecard_frame, text=cat).grid(row=i+11, column=0, padx=5, pady=2, sticky=tk.W)
            score_label = ttk.Label(scorecard_frame, text="-")
            score_label.grid(row=i+11, column=1, padx=5, pady=2)
            self.score_entries[idx] = score_label
            avail_label = ttk.Label(scorecard_frame, text="")
            avail_label.grid(row=i+11, column=2, padx=5, pady=2)
            self.available_labels[idx] = avail_label
            
    def start_game(self):
        """Start a new game."""
        # Create new game
        self.game = Yahtzee(num_games=1, device='cpu')
        self.current_round = 0
        self.current_roll = 0
        self.current_step = 0
        self.cumulative_score = 0
        self.game_started = True
        self.last_hold_mask = None
        
        # Reset scorecard display
        for label in self.score_entries.values():
            label.config(text="-")
        for label in self.available_labels.values():
            label.config(text="✓")
        self.bonus_label.config(text="-")
        
        # Enable buttons
        self.step_button.config(state=tk.NORMAL)
        self.auto_play_button.config(state=tk.NORMAL)
        
        # Initialize first round
        self.game.round[:] = 0
        self.game.roll_dice()
        self.game.turn.zero_()
        self.game.turn[:, 0] = 1
        self.current_roll = 1
        
        # Update display
        self.update_display()
        self.status_label.config(text="Game started! Step through to see agent decisions.")
        
    def next_step(self):
        """Execute the next step in the game."""
        if not self.game_started:
            return
            
        if self.current_step == 0:  # Before decision
            if self.current_roll <= 2:  # Hold decision
                # Get hold decision
                hold_mask = self.player.decide_hold(self.game)
                self.last_hold_mask = hold_mask[0]
                
                # Show hold pattern
                for i in range(5):
                    if self.last_hold_mask[i]:
                        self.hold_labels[i].config(text="HOLD", foreground="red")
                    else:
                        self.hold_labels[i].config(text="ROLL", foreground="green")
                
                self.action_label.config(text=f"Agent decides hold pattern for roll {self.current_roll}")
                self.current_step = 1
                
            else:  # Category decision (roll 3)
                # Get category decision
                category = self.player.decide_category(self.game)
                cat_idx = category[0].item()
                
                self.action_label.config(text=f"Agent chooses: {self.category_names[cat_idx]}")
                
                # Score the category
                reward = self.game.score_category(category)
                points = reward[0].item()
                self.cumulative_score += points
                
                # Update scorecard
                self.score_entries[cat_idx].config(text=str(int(points)))
                self.available_labels[cat_idx].config(text="")
                
                # Check for upper bonus
                upper_total = 0
                for i in range(6):
                    text = self.score_entries[i].cget('text')
                    if text != "-":
                        upper_total += int(text)
                if upper_total >= 63:
                    self.bonus_label.config(text="35")
                    if self.bonus_label.cget('text') == "-":
                        self.cumulative_score += 35
                
                # Move to next round
                self.current_round += 1
                if self.current_round < 13:
                    self.game.round[:] = self.current_round
                    self.current_roll = 0
                    self.current_step = 1  # Will roll on next step
                else:
                    # Game over
                    self.game_started = False
                    self.step_button.config(state=tk.DISABLED)
                    self.auto_play_button.config(state=tk.DISABLED)
                    self.status_label.config(text=f"Game Over! Final Score: {self.cumulative_score}")
                    messagebox.showinfo("Game Over", f"Final Score: {self.cumulative_score}")
                    return
                
        else:  # After decision (execute action)
            if self.current_roll == 0:  # Need to roll first dice of round
                self.game.roll_dice()
                self.game.turn.zero_()
                self.game.turn[:, 0] = 1
                self.current_roll = 1
                self.action_label.config(text=f"Rolling dice for round {self.current_round + 1}")
                # Clear hold labels
                for label in self.hold_labels:
                    label.config(text="")
                self.last_hold_mask = None
                
            elif self.current_roll <= 2:  # Execute hold and reroll
                if self.last_hold_mask is not None:
                    self.game.roll_dice(self.last_hold_mask.unsqueeze(0))
                    self.current_roll += 1
                    self.game.turn.zero_()
                    self.game.turn[:, self.current_roll - 1] = 1
                    self.action_label.config(text=f"Rerolled dice - now on roll {self.current_roll}")
                    # Clear hold labels after roll
                    for label in self.hold_labels:
                        label.config(text="")
                    
            self.current_step = 0
            
        self.update_display()
        
    def auto_play_round(self):
        """Automatically play through the current round."""
        if not self.game_started:
            return
            
        while self.game_started and self.current_round < 13:
            self.next_step()
            self.root.update()
            if self.current_roll == 0:  # Round completed
                break
                
    def update_display(self):
        """Update the display with current game state."""
        # Update round and roll info
        self.round_label.config(text=f"Round: {self.current_round + 1}/13")
        self.roll_label.config(text=f"Roll: {self.current_roll}/3")
        self.score_label.config(text=f"Total Score: {self.cumulative_score}")
        
        # Update dice display
        if self.game is not None and self.current_roll > 0:
            dice_values = torch.argmax(self.game.dice[0], dim=1) + 1
            for i, value in enumerate(dice_values.tolist()):
                self.dice_labels[i].config(text=str(value))
                # Color code based on value
                colors = ['black', 'blue', 'green', 'orange', 'red', 'purple']
                self.dice_labels[i].config(foreground=colors[value-1])
        
        # Update available categories
        if self.game is not None:
            for i in range(13):
                if i < 6:  # Upper section
                    used = (self.game.upper[0, i*7:(i+1)*7].sum() > 0).item()
                else:  # Lower section
                    used = (self.game.lower[0, i-6] > 0).item()
                    
                if used:
                    self.available_labels[i].config(text="")
                else:
                    self.available_labels[i].config(text="✓")

if __name__ == "__main__":
    root = tk.Tk()
    app = YahtzeeGUI(root)
    root.mainloop()