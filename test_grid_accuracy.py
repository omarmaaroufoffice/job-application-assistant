import cv2
import numpy as np
from PIL import Image
import json
import os
from datetime import datetime

class GridAccuracyTest:
    def __init__(self):
        self.test_output_dir = "test_results"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Test screen dimensions (can be adjusted)
        self.screen_width = 1920
        self.screen_height = 1080
        self.scale_factor = 2.0  # Simulated Retina display
        
        # Calculate grid dimensions
        self.main_cell_width = self.screen_width // 40
        self.main_cell_height = self.screen_height // 40
        
        # Initialize test canvas
        self.canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.canvas.fill(255)  # White background
        
    def create_test_grid(self):
        """Create a test grid with known coordinates"""
        # Draw main grid lines
        for i in range(41):  # 41 lines for 40 cells
            x = i * self.main_cell_width
            y = i * self.main_cell_height
            
            # Vertical lines
            cv2.line(self.canvas, (x, 0), (x, self.screen_height), (200, 200, 200), 1)
            # Horizontal lines
            cv2.line(self.canvas, (0, y), (self.screen_width, y), (200, 200, 200), 1)
            
            # Add coordinate markers at intersections
            for j in range(41):
                intersection_x = i * self.main_cell_width
                intersection_y = j * self.main_cell_height
                
                if i < 40 and j < 40:  # Don't label the last lines
                    # Calculate grid coordinate
                    first_letter = chr(65 + (i // 26))
                    second_letter = chr(65 + (i % 26))
                    coord = f"{first_letter}{second_letter}{j+1}"
                    
                    # Draw coordinate marker
                    cv2.circle(self.canvas, (intersection_x, intersection_y), 3, (0, 0, 255), -1)
                    
                    # Add coordinate label
                    if i % 5 == 0 and j % 5 == 0:  # Show labels every 5 cells to avoid clutter
                        cv2.putText(self.canvas, coord,
                                  (intersection_x + 5, intersection_y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.4,
                                  (0, 0, 0),
                                  1)
    
    def test_coordinate_accuracy(self):
        """Test coordinate accuracy with known test points"""
        test_points = [
            ("AA1", (0, 0)),
            ("AA20", (0, self.main_cell_height * 19)),
            ("BA10", (self.main_cell_width * 26, self.main_cell_height * 9)),
            ("ZZ40", (self.main_cell_width * 39, self.main_cell_height * 39))
        ]
        
        results = []
        
        for coord, expected_pos in test_points:
            # Calculate actual position
            actual_x, actual_y = self.calculate_position(coord)
            
            # Calculate error
            error_x = abs(actual_x - expected_pos[0])
            error_y = abs(actual_y - expected_pos[1])
            
            # Draw test point
            cv2.circle(self.canvas, (actual_x, actual_y), 5, (0, 255, 0), -1)
            cv2.circle(self.canvas, expected_pos, 5, (255, 0, 0), 2)
            
            # Draw line between expected and actual if they differ
            if error_x > 1 or error_y > 1:
                cv2.line(self.canvas, (actual_x, actual_y), expected_pos, (255, 0, 0), 1)
            
            # Store result
            results.append({
                'coordinate': coord,
                'expected': expected_pos,
                'actual': (actual_x, actual_y),
                'error_x': error_x,
                'error_y': error_y
            })
            
            # Add label
            cv2.putText(self.canvas,
                       f"{coord} (err: {error_x:.1f}, {error_y:.1f})",
                       (actual_x + 10, actual_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 0, 0),
                       1)
        
        return results
    
    def calculate_position(self, coord):
        """Calculate pixel position from grid coordinate"""
        # Parse coordinate
        match = coord[:2]  # Get the letters
        number = int(coord[2:])  # Get the number
        
        # Calculate column index (corrected calculation)
        first_letter = ord(match[0]) - ord('A')
        second_letter = ord(match[1]) - ord('A')
        col_index = (first_letter * 26) + second_letter
        
        # Ensure column index doesn't exceed grid width
        col_index = min(col_index, 39)  # 40x40 grid (0-39)
        
        # Calculate pixel position
        x = col_index * self.main_cell_width
        y = (number - 1) * self.main_cell_height
        
        # Ensure coordinates stay within screen bounds
        x = min(x, self.screen_width - 1)
        y = min(y, self.screen_height - 1)
        
        return (int(x), int(y))
    
    def run_accuracy_test(self):
        """Run the complete accuracy test"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create test grid
        self.create_test_grid()
        
        # Run coordinate tests
        results = self.test_coordinate_accuracy()
        
        # Add legend
        legend_height = 120
        legend = np.zeros((legend_height, self.screen_width, 3), dtype=np.uint8)
        legend.fill(255)
        
        cv2.putText(legend, "Grid Accuracy Test Results:", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(legend, "• Green Dots: Actual Positions", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(legend, "• Red Circles: Expected Positions", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(legend, f"Screen: {self.screen_width}x{self.screen_height}, Scale: {self.scale_factor}", 
                   (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Combine image with legend
        final_image = np.vstack([self.canvas, legend])
        
        # Save results
        test_results = {
            'timestamp': timestamp,
            'screen_info': {
                'width': self.screen_width,
                'height': self.screen_height,
                'scale_factor': self.scale_factor
            },
            'test_results': results
        }
        
        # Save image
        image_path = os.path.join(self.test_output_dir, f"grid_accuracy_test_{timestamp}.png")
        cv2.imwrite(image_path, final_image)
        
        # Save results JSON
        results_path = os.path.join(self.test_output_dir, f"grid_accuracy_test_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return test_results, image_path

def main():
    # Run test
    test = GridAccuracyTest()
    results, image_path = test.run_accuracy_test()
    
    # Print summary
    print("\nGrid Accuracy Test Complete!")
    print(f"Results saved to: {image_path}")
    print("\nTest Results Summary:")
    for result in results['test_results']:
        print(f"\nCoordinate: {result['coordinate']}")
        print(f"Expected: {result['expected']}")
        print(f"Actual: {result['actual']}")
        print(f"Error: X={result['error_x']:.2f}px, Y={result['error_y']:.2f}px")

if __name__ == "__main__":
    main() 