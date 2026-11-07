import math

def minimax_search(depth, index, max_turn, scores, target, count):
    if depth == target:
        count[0] += 1
        return scores[index]
    
    if max_turn:
        left = minimax_search(depth + 1, index * 2, False, scores, target, count)
        right = minimax_search(depth + 1, index * 2 + 1, False, scores, target, count)
        return max(left, right)
    else:
        left = minimax_search(depth + 1, index * 2, True, scores, target, count)
        right = minimax_search(depth + 1, index * 2 + 1, True, scores, target, count)
        return min(left, right)

def alphabeta_search(depth, index, max_turn, scores, target, alpha, beta, count):
    if depth == target:
        count[0] += 1
        return scores[index]
    
    if max_turn:
        val = float('-inf')
        for i in range(2):
            res = alphabeta_search(depth + 1, index * 2 + i, False, scores, target, alpha, beta, count)
            val = max(val, res)
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return val
    else:
        val = float('inf')
        for i in range(2):
            res = alphabeta_search(depth + 1, index * 2 + i, True, scores, target, alpha, beta, count)
            val = min(val, res)
            beta = min(beta, val)
            if alpha >= beta:
                break
        return val

def evaluate_ordering(scores_list):
    depth = int(math.log2(len(scores_list)))
    c_mm = [0]
    c_ab = [0]
    val1 = minimax_search(0, 0, True, scores_list, depth, c_mm)
    val2 = alphabeta_search(0, 0, True, scores_list, depth, float('-inf'), float('inf'), c_ab)
    print(f"Scores Set: {scores_list}")
    print(f"Minimax Value: {val1} | Leaves Evaluated: {c_mm[0]}")
    print(f"Alpha-Beta Value: {val2} | Leaves Evaluated: {c_ab[0]}")
    print("-" * 40)

def main():
    print("============================================================")
    print("LAB 5 EXECUTION RESULTS")
    print("Student Name: Abdul Moiz | Roll No: BSAIM-F25-101")
    print("============================================================")
    
    raw_scores = [3, 5, 2, 9, 0, 7]
    depth = 3
    raw_scores += [0] * (2**depth - len(raw_scores))
    
    mm_cnt = [0]
    ab_cnt = [0]
    v1 = minimax_search(0, 0, True, raw_scores, depth, mm_cnt)
    v2 = alphabeta_search(0, 0, True, raw_scores, depth, float('-inf'), float('inf'), ab_cnt)
    
    print(f"Base Minimax Value: {v1} | Leaves Evaluated: {mm_cnt[0]}")
    print(f"Base Alpha-Beta Value: {v2} | Leaves Evaluated: {ab_cnt[0]}")
    print("=" * 60)
    
    test_scores = [3, 5, 2, 9, 3, 5, 2, 9]
    evaluate_ordering(test_scores)
    evaluate_ordering(test_scores[::-1])

if __name__ == "__main__":
    main()