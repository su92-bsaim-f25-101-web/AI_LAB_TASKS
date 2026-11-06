# -*- coding: utf-8 -*-
"""LAB 1 EXECUTION SCRIPT
"""

# Task 2 Implementation
CUTOFF_SCORE = 70
REQ_GPA = 2.5
CAP_INCOME = 50000

def evaluate_eligibility(score, complete, gpa, income):
    violations = []
    if not complete:
        violations.append("documents_incomplete")
    if score < CUTOFF_SCORE:
        violations.append(f"score<{CUTOFF_SCORE}")
    if gpa < REQ_GPA:
        violations.append(f"minimum_gpa<{REQ_GPA}")
    if income > CAP_INCOME:
        violations.append(f"income_over_limit>{CAP_INCOME}")
    if violations:
        return "Hold", "failed: " + ", ".join(violations)
    return "Review", "all conditions satisfied"

test_scenarios = [
    ("T1_pass", 82, True, 3.4, 35000, "Review", "all conditions clearly met"),
    ("T2_fail_min_gpa", 80, True, 2.1, 30000, "Hold", "only new condition 1 fal"),
    ("T3_fail_income", 85, True, 3.2, 72000, "Hold", "only new condition 2 fail"),
    ("T4_boundary", 70, True, 2.5, 50000, "Review", "every threshold exactly limit"),
]

print("SYNTHETIC thresholds: score>=70, documents complete, gpa>=2.5, income<=50000\n")
success_count = 0

for name, score, complete, gpa, income, expected, why in test_scenarios:
    actual, reason = evaluate_eligibility(score, complete, gpa, income)
    is_match = actual == expected
    success_count += is_match
    print(f"{name}")
    print(f"  input    : score={score}, complete={complete}, gpa={gpa}, income={income}")
    print(f"  expected : {expected}")
    print(f"  actual   : {actual}   ({'MATCH' if is_match else 'MISMATCH'})")
    print(f"  reason   : {reason}")
    print(f"  why chosen: {why}\n")

print(f"{success_count}/{len(test_scenarios)} core tests matched expectation")
print("\nEdge case: high score, incomplete documents")
print(evaluate_eligibility(91, False, 3.8, 20000))


# Task 3 Comparison
def rule_engine(score, complete):
    return "Review" if complete and score >= 70 else "Hold"

def prob_classifier(probability):
    return "Review" if probability >= 0.70 else "Hold"

eval_cases = [
    ("complete_high", 82, True, 0.81),
    ("complete_low", 68, True, 0.74),
    ("incomplete_high", 91, False, 0.88),
    ("boundary", 70, True, 0.70),
]

print("\ncase | symbolic | threshold | disagreement")
for name, score, complete, probability in eval_cases:
    a, b = rule_engine(score, complete), prob_classifier(probability)
    print(name, a, b, a != b, sep=" | ")

print("\n--- Boundary isolation ---")
print("variant | symbolic | threshold | disagreement")
edge_variants = [
    ("boundary_score_69 (prob 0.70)", 69, True, 0.70),
    ("boundary_prob_0.69 (score 70)", 70, True, 0.69),
]

for name, score, complete, probability in edge_variants:
    a, b = rule_engine(score, complete), prob_classifier(probability)
    print(name, a, b, a != b, sep=" | ")