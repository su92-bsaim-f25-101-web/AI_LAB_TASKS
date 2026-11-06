import heapq

NET_MAP = {
    "S": [("A", 1), ("B", 4)],
    "A": [("C", 2)],
    "B": [("G", 5)],
    "C": [("G", 3)],
    "G": []
}

H_VALS = {
    "S": 5,
    "A": 4,
    "B": 4,
    "C": 2,
    "G": 0
}

INIT_NODE = "S"
TARGET_NODE = "G"

def exec_search(net, h_map, start=INIT_NODE, goal=TARGET_NODE):
    pq = [(h_map[start], 0, start, [start])]
    best_cost = {start: 0}
    proc_log = []
    exp_log = []
    
    while pq:
        f, g, node, path = heapq.heappop(pq)
        
        if g != best_cost.get(node):
            continue
            
        proc_log.append(node)
        
        if node == goal:
            return {
                "path": path,
                "cost": g,
                "processed": proc_log,
                "expanded": exp_log,
                "expansion_count": len(exp_log)
            }
            
        exp_log.append(node)
        
        for nxt, weight in net.get(node, []):
            ng = g + weight
            if ng < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = ng
                nf = ng + h_map[nxt]
                heapq.heappush(pq, (nf, ng, nxt, path + [nxt]))
                
    return None

def main():
    astar_res = exec_search(NET_MAP, H_VALS)
    
    zero_h_map = {n: 0 for n in NET_MAP}
    ucs_res = exec_search(NET_MAP, zero_h_map)
    
    bad_h_map = dict(H_VALS)
    bad_h_map["C"] = 10
    over_res = exec_search(NET_MAP, bad_h_map)
    
    print("RESULTS SUMMARY:")
    for name, res in [("A*", astar_res), ("UCS", ucs_res), ("A* with Overestimate", over_res)]:
        if res:
            print(f"{name} -> Path: {res['path']}, Cost: {res['cost']}, Expanded Count: {res['expansion_count']}")

if __name__ == "__main__":
    main()