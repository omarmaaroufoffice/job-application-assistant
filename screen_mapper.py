import tkinter as tk
import json
import os
from PIL import Image, ImageDraw
import pyautogui

class ScreenMapper:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.title("Screen Mapper")
        
        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.root, 
            width=self.screen_width, 
            height=self.screen_height, 
            bg='white'
        )
        self.canvas.pack(fill='both', expand=True)
        
        # Grid settings - 10x10 grid
        self.grid_cols = 10
        self.grid_rows = 10
        self.cell_width = self.screen_width // self.grid_cols
        self.cell_height = self.screen_height // self.grid_rows
        self.marker_size = 20  # Larger markers for better visibility
        
        # Store grid points
        self.grid_points = {}
        
        # Bind escape key to close
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        
        # Create and save the grid
        self.create_grid()
        self.save_grid_points()
        
    def create_grid(self):
        """Create a 10x10 grid with numbered cells"""
        point_num = 1
        
        for i in range(self.grid_rows + 1):
            for j in range(self.grid_cols + 1):
                x = j * self.cell_width
                y = i * self.cell_height
                
                # Draw grid lines
                if i < self.grid_rows:
                    self.canvas.create_line(
                        x, y, x, y + self.cell_height, 
                        fill='gray', width=2
                    )
                if j < self.grid_cols:
                    self.canvas.create_line(
                        x, y, x + self.cell_width, y, 
                        fill='gray', width=2
                    )
                
                # Draw corner marker
                marker = self.canvas.create_oval(
                    x - self.marker_size//2, 
                    y - self.marker_size//2,
                    x + self.marker_size//2, 
                    y + self.marker_size//2,
                    fill='red', outline='darkred', width=2
                )
                
                # Add coordinate label (e.g., "A1", "B2", etc.)
                col_letter = chr(65 + j) if j < 26 else chr(71 + j)  # A, B, C, ...
                row_num = i + 1
                coord_label = f"{col_letter}{row_num}"
                
                label = self.canvas.create_text(
                    x + self.marker_size,
                    y - self.marker_size,
                    text=coord_label,
                    fill='blue',
                    font=('Arial', 12, 'bold')
                )
                
                # Store point coordinates with both number and coordinate label
                self.grid_points[coord_label] = {
                    'x': x,
                    'y': y,
                    'point_num': point_num
                }
                
                point_num += 1
        
        # Add instructions
        instructions = """
        10x10 Screen Mapping Grid
        
        - Each red dot is a reference point
        - Labels show grid coordinates (e.g., A1, B2)
        - Cell size: {}x{} pixels
        - Press ESC to close
        
        Use these coordinates to reference screen positions.
        Example: 'A1' is top-left, 'J10' is bottom-right
        """.format(self.cell_width, self.cell_height)
        
        self.canvas.create_text(
            self.screen_width//2,
            50,
            text=instructions,
            fill='black',
            font=('Arial', 14, 'bold')
        )
    
    def save_grid_points(self):
        """Save grid points to a JSON file"""
        grid_data = {
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'cell_width': self.cell_width,
            'cell_height': self.cell_height,
            'points': self.grid_points
        }
        
        with open('grid_reference.json', 'w') as f:
            json.dump(grid_data, f, indent=2)
        
        # Also save as image
        self.root.update()
        self.save_grid_image()
    
    def save_grid_image(self):
        """Save the grid as an image"""
        # Create a new image
        img = Image.new('RGB', (self.screen_width, self.screen_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw grid lines
        for i in range(self.grid_rows + 1):
            for j in range(self.grid_cols + 1):
                x = j * self.cell_width
                y = i * self.cell_height
                
                # Draw grid lines
                if i < self.grid_rows:
                    draw.line([(x, y), (x, y + self.cell_height)], fill='gray', width=2)
                if j < self.grid_cols:
                    draw.line([(x, y), (x + self.cell_width, y)], fill='gray', width=2)
                
                # Draw point markers
                draw.ellipse(
                    [x - self.marker_size//2, 
                     y - self.marker_size//2,
                     x + self.marker_size//2, 
                     y + self.marker_size//2],
                    fill='red', outline='darkred', width=2
                )
                
                # Add coordinate labels
                if i < self.grid_rows and j < self.grid_cols:
                    col_letter = chr(65 + j) if j < 26 else chr(71 + j)
                    row_num = i + 1
                    coord_label = f"{col_letter}{row_num}"
                    draw.text((x + self.marker_size, y - self.marker_size), 
                            coord_label, fill='blue')
        
        img.save('grid_reference.png')

def get_grid_position(coord_label):
    """Get coordinates for a grid position (e.g., 'A1', 'B2')"""
    with open('grid_reference.json', 'r') as f:
        grid_data = json.load(f)
    
    if coord_label in grid_data['points']:
        point = grid_data['points'][coord_label]
        return point['x'], point['y']
    else:
        raise ValueError(f"Invalid grid coordinate: {coord_label}")

def move_to_grid_position(coord_label):
    """Move to a position using grid coordinates (e.g., 'A1', 'B2')"""
    try:
        x, y = get_grid_position(coord_label)
        print(f"Moving to position {coord_label}: ({x}, {y})")
        pyautogui.moveTo(x, y, duration=0.5)
    except Exception as e:
        print(f"Error moving to position: {e}")

if __name__ == "__main__":
    # Create the grid
    mapper = ScreenMapper()
    mapper.root.mainloop()
    
    # Example usage:
    # move_to_grid_position('A1')  # Move to top-left
    # move_to_grid_position('E5')  # Move to center
    # move_to_grid_position('J10') # Move to bottom-right 