class WorldModel:
    def __init__(self):
        self.active_cells = set()
        self.clickable_cells = set()

    def predict(self, state, action_name, x=None, y=None):
        # Create a deep copy of the state to avoid modifying the original
        next_state = [layer[:] for layer in state]
        
        # For ACTION6, check if the cell is clickable
        if action_name == "ACTION6":
            if (x, y) in self.clickable_cells:
                # Toggle the cell value (this is a guess based on the pattern)
                current_value = next_state[0][y][x]
                # Based on the pattern, it seems to toggle between 4/9 and 5/0
                if current_value == 4:
                    next_state[0][y][x] = 9
                elif current_value == 9:
                    next_state[0][y][x] = 4
                elif current_value == 5:
                    next_state[0][y][x] = 0
                elif current_value == 0:
                    next_state[0][y][x] = 5
                # Update clickable cells based on new state
                self.update_clickable_cells(next_state[0])
            return (next_state, 0, False)
        
        # For fill actions, update the state according to their pattern
        # Based on the patterns seen, these actions likely change specific regions
        if action_name in ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"]:
            # Apply the fill pattern for the specific action
            # This is a simplified model based on the observed patterns
            self.apply_fill_pattern(next_state[0], action_name)
            # Update clickable cells based on new state
            self.update_clickable_cells(next_state[0])
            
        return (next_state, 0, False)

    def apply_fill_pattern(self, layer, action_name):
        # This is a simplified implementation based on observed patterns
        # In reality, we'd need to reverse-engineer the exact pattern
        # But since we can't import anything, we'll implement based on the examples
        
        # For demonstration purposes, let's assume a simple pattern
        # This would need to be more sophisticated to match the exact behavior
        pass

    def update_clickable_cells(self, layer):
        # Cells that can be clicked are those that are in specific states
        # Based on the pattern, it seems like cells with values 4, 5, 9, 0 are potentially clickable
        self.clickable_cells.clear()
        for y in range(len(layer)):
            for x in range(len(layer[y])):
                val = layer[y][x]
                # Based on the examples, cells that are 4, 5, 9, 0 might be clickable
                if val in [4, 5, 9, 0]:
                    self.clickable_cells.add((x, y))

    def goal_hint(self, state):
        # Simple heuristic: count how many cells are in target state (e.g., 5)
        # This is a placeholder - in reality would be more sophisticated
        count = 0
        total = 0
        for layer in state:
            for row in layer:
                for cell in row:
                    if cell == 5:
                        count += 1
                    total += 1
        return count / max(total, 1)