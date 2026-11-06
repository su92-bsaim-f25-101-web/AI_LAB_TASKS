from collections import deque
import heapq

base_graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": ["G"],
    "F": [],
    "G": []
}

def execute_bfs(start_node, target_node):
    queue = deque([(start_node, [start_node])])
    visited_set = {start_node}
    traversal_order = []
    
    while queue:
        current_node, current_path = queue.popleft()
        traversal_order.append(current_node)
        
        if current_node == target_node:
            return current_path, traversal_order
            
        for neighbor in base_graph[current_node]:
            if neighbor not in visited_set:
                visited_set.add(neighbor)
                queue.append((neighbor, current_path + [neighbor]))
    return None, traversal_order


def execute_dfs(start_node, target_node):
    stack = [(start_node, [start_node])]
    visited_set = set()
    traversal_order = []
    
    while stack:
        current_node, current_path = stack.pop()
        
        if current_node in visited_set:
            continue
        visited_set.add(current_node)
        traversal_order.append(current_node)

        if current_node == target_node:
            return current_path, traversal_order

        for neighbor in reversed(base_graph[current_node]):
            stack.append((neighbor, current_path + [neighbor]))
            
    return None, traversal_order


def depth_limited_search(node, goal, depth_limit, current_path):
    if node == goal:
        return "FOUND", current_path
    if depth_limit == 0:
        if base_graph[node]:
            return "CUTOFF", None
        return "FAILURE", None
        
    is_cutoff = False
    for child in base_graph[node]:
        if child in current_path:
            continue
        status, result = depth_limited_search(child, goal, depth_limit - 1, current_path + [child])
        if status == "FOUND":
            return status, result
        if status == "CUTOFF":
            is_cutoff = True
            
    if is_cutoff:
        return "CUTOFF", None
    return "FAILURE", None


def iterative_deepening_dfs(start, goal, max_depth_limit):
    for limit in range(max_depth_limit + 1):
        status, path = depth_limited_search(start, goal, limit, [start])
        print(f"Limit: {limit} | Status: {status} | Path: {path}")
        if status == "FOUND":
            return path
    return None


weighted_graph = {
    "A": [("B", 1), ("C", 2)],
    "B": [("D", 4), ("E", 1)],
    "C": [("F", 5), ("G", 10), ("G", 12)],
    "D": [],
    "E": [("G", 1)],
    "F": [],
    "G": []
}

def uniform_cost_search(start_node, goal_node):
    priority_queue = [(0, start_node, [start_node])]
    shortest_costs = {start_node: 0}
    expansion_order = []
    
    while priority_queue:
        current_cost, current_node, current_path = heapq.heappop(priority_queue)
        
        if current_cost != shortest_costs.get(current_node):
            continue
            
        expansion_order.append(current_node)
        
        if current_node == goal_node:
            return current_path, current_cost, expansion_order
            
        for neighbor, edge_weight in weighted_graph[current_node]:
            updated_cost = current_cost + edge_weight
            if updated_cost < shortest_costs.get(neighbor, float("inf")):
                shortest_costs[neighbor] = updated_cost
                heapq.heappush(priority_queue, (updated_cost, neighbor, current_path + [neighbor]))
                
    return None, float("inf"), expansion_order


if __name__ == "__main__":
    print("BFS:", execute_bfs("A", "G"))
    print("DFS:", execute_dfs("A", "G"))
    
    for limit_val in [1, 2, 3, 4]:
        print(f"DLS limit {limit_val}:", depth_limited_search("A", "G", limit_val, ["A"]))
        
    iterative_deepening_dfs("A", "G", 4)
    
    path, cost, order = uniform_cost_search("A", "G")
    print("UCS Path:", path)
    print("Total Cost:", cost)
    print("Processing Order:", order)