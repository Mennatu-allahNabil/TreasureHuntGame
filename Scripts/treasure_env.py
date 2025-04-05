import numpy as np
import random
import gym
from gym import spaces
from enum import Enum
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QFont, QBrush, QPen, QIcon


# Define all the cell types in the grid
class CellType(Enum):
    EMPTY, TREASURE, TRAP, OBSTACLE, ENEMY, SPECIAL = range(6)  # Enum values from 0 to 5


"""Draw the grid and agent"""
class GameCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # Initialize parent class
        self.grid, self.size, self.agent_pos = None, 0, (0, 0)  # Initialize grid, size and agent position
        self.cell_size = 40  # Cell size in pixels
        self.setMinimumSize(400, 400)  # Set minimum size for the widget

    """Update the game to display"""
    def set_data(self, grid, size, agent_pos):
        self.grid, self.size, self.agent_pos = grid, size, agent_pos
        self.calculate_cell_size()  # Calculate cell size based on new dimensions
        self.update()  # Trigger repaint

    """Update the cell size based on widget dimensions and grid size"""
    def calculate_cell_size(self):
        if self.size <= 0: return
        self.cell_size = min(self.width() // self.size, self.height() // self.size)

    """Adjust cell size when resizing"""
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.calculate_cell_size()

    """Paint the grid, items, and agent"""
    def paintEvent(self, event):
        if self.grid is None: return
        painter = QPainter(self)

        # Calculate offset to center the grid in the widget
        x_offset = (self.width() - (self.size * self.cell_size)) // 2
        y_offset = (self.height() - (self.size * self.cell_size)) // 2

        # Cell colors and emojis for each cell type
        cell_colors = {
            CellType.EMPTY.value: QColor("#F5F5DC"),  # Beige color for empty cells
            CellType.TREASURE.value: QColor("#2A4500"),  # Dark green for treasure
            CellType.TRAP.value: QColor("#C24500"),  # Orange-red for traps
            CellType.OBSTACLE.value: QColor("#d4d3d6"),  # Light gray for obstacles
            CellType.ENEMY.value: QColor("#7621A3"),  # Purple for enemies
            CellType.SPECIAL.value: QColor("#00C2FF")  # Light blue for special treasures
        }

        cell_emojis = {
            CellType.EMPTY.value: "",  # No emoji for empty cells
            CellType.TREASURE.value: "💰",  # Money bag for treasure
            CellType.TRAP.value: "🔥",  # Fire for traps
            CellType.OBSTACLE.value: "🧱",  # Brick for obstacles
            CellType.ENEMY.value: "💀",  # Skull for enemies
            CellType.SPECIAL.value: "💎"  # Gem for special treasures
        }

        # Draw grid cells
        for x in range(self.size):
            for y in range(self.size):
                x1, y1 = x_offset + y * self.cell_size, y_offset + x * self.cell_size
                cell_type = self.grid[x][y]
                emoji_size = 20

                # Draw cell background with color based on cell type
                painter.setBrush(QBrush(cell_colors[cell_type]))
                painter.setPen(QPen(QColor("gray")))
                painter.drawRect(x1, y1, self.cell_size, self.cell_size)

                # Draw cell emoji based on the cell type
                emoji = cell_emojis[cell_type]
                if emoji:
                    painter.setFont(QFont("Arial", emoji_size))
                    painter.drawText(x1, y1, self.cell_size, self.cell_size, Qt.AlignCenter, emoji)

                # Draw agent at current position
                if (x, y) == self.agent_pos:
                    painter.setFont(QFont("Arial", emoji_size))
                    painter.drawText(x1, y1, self.cell_size, self.cell_size, Qt.AlignCenter, "🧙")


"""Gym environment for the Treasure Hunt game"""
class TreasureHuntEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    # Action space constants
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, size=9, max_steps=100, lives=3, special_duration=10, render_mode=None):
        super().__init__()
        # Game settings
        self.size = size  # Grid size (size x size)
        self.max_steps = max_steps  # Maximum steps the agent can take before game ends
        self.lives = lives  # Starting number of lives that agent can consume before game ends
        self.lives_reset = lives  # Starting number of lives that agent can consume before game ends
        self.special_duration = special_duration  # How long special treasures last on the grid
        self.render_mode = render_mode  # Rendering mode

        # Action space: available actions for the agent on the grid (removed COLLECT)
        self.action_space = spaces.Discrete(4)  # Up, Right, Down, Left

        # Observation space: grid + agent position + steps left + lives + special treasures
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=5, shape=(size, size), dtype=np.int32),
            'agent_pos': spaces.MultiDiscrete([size, size]),
            'steps_left': spaces.Discrete(max_steps + 1),
            'lives': spaces.Discrete(lives + 1),
            'special_treasures': spaces.Box(low=0, high=max_steps, shape=(size * size, 2), dtype=np.int32)
        })

        # Initialize rendering components when render mode is human
        self.app = None
        self.window = None
        if self.render_mode == 'human':
            self.app = QApplication.instance()
            if not self.app:
                self.app = QApplication(sys.argv)
            self.window = TreasureHuntUI(size=size, max_steps=max_steps, special_duration=special_duration)

        # Initialize state variables
        self.reset()
 """Reset the environment to initial state"""
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Reset game state
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)  # Empty grid
        self.agent_pos = (self.size//2, self.size//2)  # Start agent in center
        self.steps_left = self.max_steps  # Reset step counter
        self.score = 0  # Reset score
        self.lives = self.lives_reset  # Reset lives
        self.game_over = False  # Game not over yet
        self.special_treasures = []  # No special treasures at start
        self.steps_to_next_special = random.randint(15, 30)  # Random steps until first special treasure appears on the grid
        self.collectible_cells = [CellType.TREASURE.value, CellType.SPECIAL.value]  # Define collectible items as both the normal treasuers and the special treasures
        
        # Track rewards by type
        self.reward_by_type = {
            "treasure": 0,
            "special": 0,
            "trap": 0,
            "enemy": 0,
            "timeout": 0,
            "all_treasures": 0
        }

        # Create the grid
        self._create_map()

        # Update rendering if active
        if self.render_mode == 'human' and self.window:
            self.window.grid = self.grid.copy()
            self.window.agent_pos = self.agent_pos
            self.window.steps_left = self.steps_left
            self.window.lives = self.lives
            self.window.score = self.score
            self.window.special_treasures = self.special_treasures.copy()
            self.window.update_display()
            self.window.show()

        return self._get_observation(), {}

    """Get the current observation state"""
    def _get_observation(self):
        # Convert special treasures to numpy array format for observation space
        special_array = np.zeros((self.size * self.size, 2), dtype=np.int32)
        for i, ((x, y), duration) in enumerate(self.special_treasures):
            if i < self.size * self.size:  # Check if the number of special treasures is below the grid cells number
                special_array[i] = [x * self.size + y, duration]
        return {
            'grid': self.grid.copy(),
            'agent_pos': np.array(self.agent_pos),
            'steps_left': self.steps_left,
            'lives': self.lives,
            'special_treasures': special_array
        }

    """Create the grid with random placement of items"""
    def _create_map(self):
        self.grid = np.array([
            [CellType.OBSTACLE.value, CellType.TREASURE.value, CellType.OBSTACLE.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.OBSTACLE.value],
            [CellType.TRAP.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.TREASURE.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.TRAP.value, CellType.EMPTY.value, CellType.EMPTY.value],
            [CellType.OBSTACLE.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.TREASURE.value, CellType.EMPTY.value, CellType.OBSTACLE.value, CellType.EMPTY.value, CellType.TREASURE.value, CellType.TRAP.value],
            [CellType.EMPTY.value, CellType.EMPTY.value, CellType.TREASURE.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.OBSTACLE.value, CellType.EMPTY.value],
            [CellType.EMPTY.value, CellType.EMPTY.value, CellType.TREASURE.value, CellType.OBSTACLE.value, CellType.EMPTY.value, CellType.TREASURE.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.EMPTY.value],
            [CellType.EMPTY.value, CellType.TREASURE.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.TREASURE.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.EMPTY.value],
            [CellType.OBSTACLE.value, CellType.EMPTY.value, CellType.ENEMY.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.OBSTACLE.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.EMPTY.value],
            [CellType.EMPTY.value, CellType.TRAP.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.TREASURE.value, CellType.TRAP.value, CellType.TREASURE.value, CellType.EMPTY.value, CellType.OBSTACLE.value],
            [CellType.EMPTY.value, CellType.TREASURE.value, CellType.EMPTY.value, CellType.TRAP.value, CellType.TREASURE.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.EMPTY.value, CellType.EMPTY.value]
        ])
        
    
    """Perform step in the environment"""
    def step(self, action):
        if self.game_over or self.steps_left <= 0:
            return self._get_observation(), 0, True, False, {'event': 'game_over', 'rewards_by_type': self.reward_by_type}

        event = ""
        reward = 0  # Base step penalty
        
        # Initialize tracking variables if they don't exist
        if not hasattr(self, 'previous_positions'):
            self.previous_positions = []  # Track previous positions
        if not hasattr(self, 'position_counts'):
            self.position_counts = {}  # Count visits to each position
        if not hasattr(self, 'steps_since_danger'):
            self.steps_since_danger = 0  # Count steps since last trap or enemy
        if not hasattr(self, 'missed_special'):
            self.missed_special = []  # Track special treasures that disappeared

        old_pos = self.agent_pos
        moved = False

        # Movement actions in the 4-neighbours
        if action < 4:  # UP, RIGHT, DOWN, LEFT
            x, y = self.agent_pos
            dx, dy = {
                self.UP: (-1, 0),
                self.RIGHT: (0, 1),
                self.DOWN: (1, 0),
                self.LEFT: (0, -1)
            }[action]

            nx, ny = x + dx, y + dy
            # Move agent if valid
            if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[nx][ny] != CellType.OBSTACLE.value:
                self.agent_pos = (nx, ny)
                moved = True

                # Automatic collection of treasures when stepping on them
                cell_type = self.grid[nx, ny]
                if cell_type in self.collectible_cells:
                    if cell_type == CellType.TREASURE.value:
                        reward += 30
                        self.score += 30
                        self.reward_by_type["treasure"] += 30
                        self.grid[nx, ny] = CellType.EMPTY.value
                        event = "treasure_collected"
                    elif cell_type == CellType.SPECIAL.value:
                        reward += 50
                        self.score += 50
                        self.reward_by_type["special"] += 50
                        self.grid[nx, ny] = CellType.EMPTY.value
                        # Remove from special treasures list
                        self.special_treasures = [t for t in self.special_treasures if t[0] != (nx, ny)]
                        event = "special_collected"

                # Handle traps and enemies
                elif cell_type == CellType.TRAP.value:
                    reward -= 10
                    self.score -= 10
                    self.reward_by_type["trap"] -= 10
                    event = "trapped"
                    self.steps_since_danger = 0  # Reset the counter when hitting a trap
                elif cell_type == CellType.ENEMY.value:
                    self.lives -= 1
                    reward -= 20
                    self.score = self.score - 20
                    self.reward_by_type["enemy"] -= 20
                    event = "caught"
                    self.steps_since_danger = 0  # Reset the counter when hitting an enemy
                    if self.lives <= 0:
                        self.game_over = True
        
        # Track positions
        self.previous_positions.append(self.agent_pos)
        if len(self.previous_positions) > 20:  # Keep only last 20 positions
            self.previous_positions.pop(0)
        
        # Count position visits
        pos_key = str(self.agent_pos)
        self.position_counts[pos_key] = self.position_counts.get(pos_key, 0) + 1
        
        # PENALTY 1: Penalize for staying in the same position (-2)
        if not moved:
            reward -= 2
            self.score -= 2
            if not hasattr(self.reward_by_type, "stuck"):
                self.reward_by_type["stuck"] = 0
            self.reward_by_type["stuck"] -= 2
            event += " stuck"
            # Reset danger avoidance counter when stuck
            self.steps_since_danger = 0
        
        # PENALTY 2: Penalize for revisiting the same 4 or fewer cells multiple times (-5)
        if len(set(self.previous_positions[-10:])) <= 4 and len(self.previous_positions) >= 10:
            # Check if we've moved between these few cells at least 3 times
            if self.position_counts[pos_key] >= 3:
                reward -= 5
                self.score -= 5
                if not hasattr(self.reward_by_type, "circle_movement"):
                    self.reward_by_type["circle_movement"] = 0
                self.reward_by_type["circle_movement"] -= 5
                event +=" circle_movement"
        
        # PENALTY 3: Penalize for missing special rewards (-3)
        # Check if any special treasures disappeared in this step
        old_special_positions = set(pos for pos, _ in self.special_treasures)
        
        # Update special treasures
        old_special_treasures = self.special_treasures.copy()
        self.update_special_treasures()
        
        # Find disappeared special treasures (excluding collected ones)
        new_special_positions = set(pos for pos, _ in self.special_treasures)
        disappeared = old_special_positions - new_special_positions
        
        # If the agent didn't collect it but it disappeared, penalize
        for pos in disappeared:
            if pos != self.agent_pos:  # It wasn't collected, it timed out
                reward -= 3
                self.score -= 3
                if not hasattr(self.reward_by_type, "missed_special"):
                    self.reward_by_type["missed_special"] = 0
                self.reward_by_type["missed_special"] -= 3
                self.missed_special.append(pos)
                event += " missed_special"
        
        # BONUS: Reward for avoiding traps and enemies for 10 steps (+10)
        # Only increment if the agent actually moved
        if moved:
            self.steps_since_danger += 1
            if self.steps_since_danger >= 10:
                # Only give bonus once every 10 steps
                if self.steps_since_danger % 10 == 0:
                    reward += 10
                    self.score += 10
                    if not hasattr(self.reward_by_type, "danger_avoidance"):
                        self.reward_by_type["danger_avoidance"] = 0
                    self.reward_by_type["danger_avoidance"] += 10
                    event +=  " danger_avoidance_bonus"
        
        # Update the steps left for the agent  
        self.steps_left -= 1

        # Special treasure spawning logic
        self.steps_to_next_special -= 1
        if self.steps_to_next_special <= 0:
            self.spawn_special_treasure()
            self.steps_to_next_special = random.randint(15, 30)

        # Check ending conditions
        done = False
        if self.steps_left <= 0:
            self.reward_by_type["timeout"] -= 200
            reward -= 200
            self.game_over = True
            event = "timeout"
            done = True

        if self.lives <= 0:
            self.game_over = True
            event = "game_over"
            done = True

        # Check if all treasures collected
        if not np.any(self.grid == CellType.TREASURE.value) and not done:
            reward += 200  # Bonus for collecting all treasures
            self.score += 200
            self.reward_by_type["all_treasures"] += 200
            self.game_over = True
            event = "all_treasures_collected"
            done = True

        # Update rendering if active
        if self.render_mode == 'human' and self.window:
            self.window.grid = self.grid.copy()
            self.window.agent_pos = self.agent_pos
            self.window.steps_left = self.steps_left
            self.window.lives = self.lives
            self.window.score = self.score
            self.window.special_treasures = self.special_treasures.copy()
            self.window.update_display()
            self.window.event_label.setText(f"Event: {event}" if event else "")

        return self._get_observation(), reward, done, False, {'event': event, 'score': self.score, 'rewards_by_type': self.reward_by_type}
    """Spawn a special treasure that disappears after a time"""
    def spawn_special_treasure(self):
        # Find empty cells where special treasures can be placed
        empty_cells = [(x, y) for x in range(self.size) for y in range(self.size)
                      if self.grid[x][y] == CellType.EMPTY.value and (x, y) != self.agent_pos]

        if empty_cells:
            x, y = random.choice(empty_cells)
            self.grid[x][y] = CellType.SPECIAL.value
            self.special_treasures.append([(x, y), self.special_duration])
            return True
        return False

    """Update timers for special treasures and remove expired ones"""
    def update_special_treasures(self):
        updated_treasures = []
        for (x, y), duration in self.special_treasures:
            if duration > 1:
                updated_treasures.append([(x, y), duration - 1])
            elif self.grid[x][y] == CellType.SPECIAL.value:
                self.grid[x][y] = CellType.EMPTY.value

        self.special_treasures = updated_treasures

    """Render the environment"""
    def render(self):
        if self.render_mode == 'human':
            if self.window:
                self.app.processEvents()
            return None


    """Close the environment and cleanup resources"""
    def close(self):
        if self.window:
            self.window.close()
            self.window = None


class TreasureHuntUI(QMainWindow):
    """Main window for the Treasure Hunt game"""
    def __init__(self, size=25, max_steps=150, special_duration=10):
        super().__init__()
        # Game settings
        self.size, self.max_steps, self.special_duration = size, max_steps, special_duration
        self.grid = np.zeros((size, size), dtype=np.int32)  # Initialize empty grid
        self.agent_pos = (size//2, size//2)  # Start agent in center
        self.steps_left = max_steps  # Initialize step counter
        self.score, self.lives = 0, 3  # Initialize score and lives
        self.game_over = False  # Game not over yet
        self.special_treasures = []  # No special treasures at start
        self.steps_to_next_special = random.randint(15, 30)  # Random steps until first special treasure appears on the grid

        # Setup UI components
        self.setup_ui()
        self.setWindowTitle("Treasure Hunt Game")
        self.resize(600, 700)

    """Setup the ui components"""
    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Game canvas for drawing the grid
        self.canvas = GameCanvas()
        main_layout.addWidget(self.canvas, 1)

        # Info display frame for game stats
        info_frame = QFrame()
        info_layout = QVBoxLayout(info_frame)
        self.info_label = QLabel()  # Display score, lives, steps
        self.info_label.setFont(QFont("Arial", 12))
        self.special_label = QLabel()  # Display special treasures info
        self.special_label.setFont(QFont("Arial", 10))
        self.event_label = QLabel("Gym Environment Running")  # Display event messages
        self.event_label.setFont(QFont("Arial", 10))

        info_layout.addWidget(self.info_label)
        info_layout.addWidget(self.special_label)
        info_layout.addWidget(self.event_label)
        main_layout.addWidget(info_frame)

    """Update the display with current game state"""
    def update_display(self):
        self.canvas.set_data(self.grid, self.size, self.agent_pos)

        # Update info labels
        heart_icons = "❤️" * self.lives
        self.info_label.setText(f"Score: {self.score} | Lives: {heart_icons} | Steps left: {self.steps_left}")

        # Update special treasures info
        if self.special_treasures:
            special_info = [f"💎 at ({y},{x}): {time} steps left" for (x, y), time in self.special_treasures]
            self.special_label.setText(" | ".join(special_info))
        else:
            self.special_label.setText("No special treasures active")
