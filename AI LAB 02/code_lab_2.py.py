# -*- coding: utf-8 -*-

class SmartClimateController:
    def __init__(self):
        self.last_mode = None

    def evaluate_state(self, temperature, is_occupied):
        if not is_occupied:
            desired_mode = "ECO"
        elif temperature > 26:
            desired_mode = "COOL"
        elif temperature < 20:
            desired_mode = "WARM"
        else:
            desired_mode = "IDLE"

        execute_action = (desired_mode != self.last_mode)
        previous_tracked = self.last_mode
        self.last_mode = desired_mode
        
        return previous_tracked, desired_mode, execute_action


if __name__ == "__main__":
    controller = SmartClimateController()
    
    test_percepts = [
        (29, True),
        (29, True),
        (20, True),
        (26, True),
        (19, True),
        (29, False),
        (29, False)
    ]

    print("Step | Last State    | Percept      | Desired Mode | Action Status")
    print("-" * 65)
    
    for step_num, percept_data in enumerate(test_percepts, start=1):
        prev_state, target_mode, trigger_flag = controller.evaluate_state(*percept_data)
        status_label = "SEND COMMAND" if trigger_flag else "NO COMMAND"
        print(f"{step_num:4d} | {str(prev_state):13s} | {str(percept_data):12s} | {target_mode:12s} | {status_label}")