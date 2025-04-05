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
class TreasureHuntUI(QMainWindow):
    pass
