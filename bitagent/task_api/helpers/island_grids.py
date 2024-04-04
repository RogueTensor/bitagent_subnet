import random
import numpy as np

def can_place_island(grid, start_row, start_col, island_shape):
    """
    Check if an island of a given shape can be placed at the start position without overlapping other islands.
    """
    for row_offset, col_offset in island_shape:
        new_row = start_row + row_offset
        new_col = start_col + col_offset
        if new_row < 0 or new_row >= len(grid) or new_col < 0 or new_col >= len(grid[0]) or grid[new_row][new_col] != 0:
            return False

        # Check adjacent cells with added boundary checks
        if (new_row + 1 < len(grid) and grid[new_row+1][new_col] != 0) or \
           (new_row - 1 >= 0 and grid[new_row-1][new_col] != 0) or \
           (new_col + 1 < len(grid[0]) and grid[new_row][new_col+1] != 0) or \
           (new_col - 1 >= 0 and grid[new_row][new_col-1] != 0):
            return False
    return True

def place_island(grid, start_row, start_col, island_shape):
    """
    Place an island on the grid at the specified starting position.
    """
    for row_offset, col_offset in island_shape:
        grid[start_row + row_offset][start_col + col_offset] = 1

def generate_island_shapes(max_size, num_shapes):
    """
    Generate a list of possible island shapes within the max_size constraint.
    Each shape is a list of (row_offset, col_offset) tuples from the starting cell.
    """
    # countering the effect of setting seed for task orchestration from validators
    random.seed(None)
    shapes = []
    for _ in range(num_shapes):
        size = random.randint(1, max_size)
        shape = [(0, 0)]  # Start with the initial cell
        for __ in range(size - 1):
            # Add a new cell adjacent to an existing cell in the shape
            adj = random.choice(shape)
            offsets = [(adj[0] + 1, adj[1]), (adj[0] - 1, adj[1]), (adj[0], adj[1] + 1), (adj[0], adj[1] - 1)]
            new_cell = random.choice(offsets)
            if new_cell not in shape:
                shape.append(new_cell)
        shapes.append(shape)
    return shapes

def generate_island_grid(num_islands, grid_size):
    """
    Generate a grid with a specified number of islands and grid size.
    """
    # countering the effect of setting seed for task orchestration from validators
    random.seed(None)
    height, width = grid_size
    grid = np.zeros((height, width), dtype=int)
    island_shapes = generate_island_shapes(min(height, width) // 4, num_islands)  # Generate island shapes

    placed_islands = 0
    attempts = 0
    max_attempts = 1000  # Prevent infinite loops

    while placed_islands < num_islands and attempts < max_attempts:
        for shape in island_shapes:
            start_row = random.randint(0, height - 1)
            start_col = random.randint(0, width - 1)
            if can_place_island(grid, start_row, start_col, shape):
                place_island(grid, start_row, start_col, shape)
                placed_islands += 1
                break  # Move to the next island shape after successful placement
            attempts += 1

    if placed_islands != num_islands:
        print("Could not place all islands within the grid. Try adjusting the grid size or number of islands.")
    return grid.tolist()
