"""
Flow Free Puzzle Image Parser (OpenCV)

This script extracts the following info from a Flow Free puzzle:
    - Grid size
    - Cell layout
    - Colored dots
    - Color pair groupings

Outputs a matrix to be fed directly into an AI / ML solver
Matrix is sorted by (row, column) and uses integers to represent different colors
"""

import cv2
import numpy as np

# Utility Function - Removes duplicate detected grid lines by clustering nearby values
# input - actual detected lines and customizable minimum spacing
def remove_duplicate_lines(lines, min_spacing):
    if not lines:
        return []

    lines = sorted(lines)
    unique = [lines[0]]

    # Iterate through sorted lines and only keep those that are sufficiently spaced apart
    for line in lines[1:]:
        if line - unique[-1] > min_spacing:
            unique.append(line)

    return unique


# Grid Detection - Finds the bounding box of the Flow Free grid
# input - original image
# output - bounding box of the grid (x, y, width, height)
# **UPDATED 3/17/26**
def find_grid_bbox(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Increase threshold to ignore the dimmer background glow
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY) # grayscale thresholding
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # contour detection
    
    # Filter for large contours (find the largest square in the image as this must be the grid)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 0.9 <= aspect_ratio <= 1.1 and w > (image.shape[1] * 0.5):
            return x, y, w, h
    
    # Fallback to original method if no square is found
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


# Grid Line Detection - Detects horizontal and vertical grid lines using edge detection
# input - grayscale image of the grid area
# output - lists of detected horizontal and vertical line positions (in pixel coordinates)
def detect_grid_lines(gray_grid):
    edges = cv2.Canny(gray_grid, 50, 150) # canny edge detection
    lines = cv2.HoughLinesP( # Hough Transform
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min(gray_grid.shape) // 2,
        maxLineGap=10
    )

    horizontal = []
    vertical = []

    if lines is None:
        return [], []

    # Classify lines as horizontal or vertical based on their angle
    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > dy:
            horizontal.append((y1 + y2) // 2)
        else:
            vertical.append((x1 + x2) // 2)

    return horizontal, vertical


# Dot Detection - Detects colored dots inside each grid cell
# input - original image, grid bounding box, number of rows and columns
# output - list of detected dots with their positions and average colors
def detect_dots(image, bbox, rows, cols):
    x, y, w, h = bbox
    cell_h = h // rows
    cell_w = w // cols

    dots = []

    for r in range(rows):
        for c in range(cols):
            cy = y + r * cell_h
            cx = x + c * cell_w

            cell = image[cy:cy + cell_h, cx:cx + cell_w]

            hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

            # Mask for bright / saturated colors
            mask = cv2.inRange(
                hsv,
                np.array([0, 80, 80]),
                np.array([180, 255, 255])
            )

            dot_pixels = cv2.countNonZero(mask)
            cell_pixels = cell_h * cell_w

            if dot_pixels > 0.1 * cell_pixels:
                avg_color = cv2.mean(cell, mask=mask)[:3]
                dots.append({
                    'position': (r, c),
                    'color': avg_color
                })

    return dots


# Color Grouping - Groups dots by color similarity
# input - list of detected dots with their positions and average colors
# output - list of color groups, each containing the average color and positions of dots in that group
def group_dots_by_color(dots):
    color_groups = []

    for dot in dots:
        matched = False
        # Compare dot color to existing groups and assign to the closest one if within threshold
        for group in color_groups:
            dist = np.linalg.norm(
                np.array(dot['color']) - np.array(group['color'])
            )
            if dist < 50:
                group['positions'].append(dot['position'])
                matched = True
                break
        # If no existing group is close enough, create a new group for this color
        if not matched:
            color_groups.append({
                'color': dot['color'],
                'positions': [dot['position']]
            })

    return color_groups


# Main Extraction Function
def extract_flow_free_puzzle(image_path):

    image = cv2.imread(image_path)

    # 1. Find grid bounding box
    x, y, w, h = find_grid_bbox(image)

    gray_grid = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    # 2. Detect grid lines
    h_lines, v_lines = detect_grid_lines(gray_grid)

    h_lines = remove_duplicate_lines(h_lines, h // 10)
    v_lines = remove_duplicate_lines(v_lines, w // 10)

    rows = len(h_lines) - 1
    cols = len(v_lines) - 1

    # 3. Detect dots
    dots = detect_dots(image, (x, y, w, h), rows, cols)

    # 4. Group dots by color
    color_groups = group_dots_by_color(dots)

    # 5. Build matrix output
    matrix = np.zeros((rows, cols), dtype=int)
    colors = {}

    for idx, group in enumerate(color_groups, start=1):
        for r, c in group['positions']:
            matrix[r][c] = idx

        colors[idx] = {
            'positions': group['positions']
        }

    return {
        'grid_size': (rows, cols),
        'matrix': matrix,
        'colors': colors
    }


# Example Usage
if __name__ == "__main__":
    result = extract_flow_free_puzzle("flowshare (level 18).png") # image path here

    #Formatted output
    print(f"Grid Size: {result['grid_size'][0]} x {result['grid_size'][1]}")
    print("\nPuzzle Matrix:")
    print(result['matrix'])

    # Color pair groupings
    print("\nColor Pairs:")
    for cid, info in result['colors'].items():
        print(f"Color {cid}: {info['positions']}")