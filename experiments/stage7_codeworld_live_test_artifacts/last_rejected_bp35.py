class WorldModel:
    def __init__(self):
        self.layer_counts = []
        self.action_sequence = []
        
    def predict(self, state, action_name, x=None, y=None):
        # Create a deep copy of the state
        next_state = [layer[:] for layer in state]
        
        # Initialize return values
        levels_delta = 0
        done = False
        
        # Store action for tracking
        self.action_sequence.append(action_name)
        
        # Process each action according to the transcript patterns
        if action_name == "ACTION1":
            # ACTION1: Changes layer count from 2 to 5
            # Add 3 empty layers
            while len(next_state) < 5:
                next_state.append([[0 for _ in range(64)] for _ in range(64)])
                
        elif action_name == "ACTION2":
            # ACTION2: Changes layer count from 5 to 4
            # Remove 1 layer
            while len(next_state) > 4:
                next_state.pop()
                
        elif action_name == "ACTION3":
            # ACTION3: Changes layer count from 4 to 3
            # Remove 1 layer
            while len(next_state) > 3:
                next_state.pop()
                
        elif action_name == "ACTION4":
            # ACTION4: Changes layer count from 3 to 5
            # Add 2 empty layers
            while len(next_state) < 5:
                next_state.append([[0 for _ in range(64)] for _ in range(64)])
                
        elif action_name == "ACTION5":
            # ACTION5: Changes layer count from 5 to 2
            # Remove 3 layers
            while len(next_state) > 2:
                next_state.pop()
                
        elif action_name == "ACTION6":
            # ACTION6: Sets specific cells to 15 based on coordinates
            # From transcript, it appears to set positions (0, y+6, 63) and (1, y+6, 63) to 15
            if x is not None and y is not None:
                # For ACTION6(0,0) - sets (0,6,63) and (1,6,63) to 15
                # For ACTION6(31,31) - sets (0,37,63) and (1,37,63) to 15  
                # For ACTION6(63,63) - sets (0,69,63) and (1,69,63) to 15 (but y+6 exceeds bounds)
                # Actually looking more carefully, it seems to be setting (0, y+6, 63) and (1, y+6, 63)
                # But when y=63, y+6=69 which is out of bounds. Let me check the actual pattern.
                # Looking at transcript, ACTION6(36,8) sets (0,14,63) and (1,14,63) to 15
                # So it's (0, y+6, 63) and (1, y+6, 63) where y=8, so 8+6=14
                # But also ACTION6(62,34) sets (0,40,63) and (1,40,63) to 15
                # So it's (0, y+6, 63) and (1, y+6, 63) where y=34, so 34+6=40
                
                offset_y = y + 6
                if offset_y < 64:
                    if len(next_state) > 0 and 63 < 64:
                        next_state[0][offset_y][63] = 15
                    if len(next_state) > 1 and 63 < 64:
                        next_state[1][offset_y][63] = 15
                        
        elif action_name == "ACTION7":
            # ACTION7: Changes layer count from 2 to 5
            # Add 3 empty layers
            while len(next_state) < 5:
                next_state.append([[0 for _ in range(64)] for _ in range(64)])
            
        return (next_state, levels_delta, done)
    
    def goal_hint(self, state):
        # Count how many cells have value 15 (which seems to indicate progress)
        count = 0
        for layer in state:
            for row in layer:
                for cell in row:
                    if cell == 15:
                        count += 1
        # Normalize by total cells to get a value between 0 and 1
        total_cells = len(state) * len(state[0]) * len(state[0][0]) if state else 1
        return count / total_cells if total_cells > 0 else 0.0