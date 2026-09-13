from typing import List, Dict, Any, Tuple
import math

def calculate_pearson_correlation(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    den = den_x * den_y
    return round(num / den, 4) if den != 0 else 0.0

def calculate_spearman_correlation(x: List[float], y: List[float]) -> float:
    def rank(arr):
        sorted_indices = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        for r, i in enumerate(sorted_indices, 1):
            ranks[i] = float(r)
        return ranks
    rx = rank(x)
    ry = rank(y)
    return calculate_pearson_correlation(rx, ry)

def calculate_cohen_kappa(cat1: List[int], cat2: List[int], categories: List[int] = [1, 2, 3, 4, 5]) -> float:
    n = len(cat1)
    if n == 0:
        return 0.0
    po = sum(1 for c1, c2 in zip(cat1, cat2) if c1 == c2) / n
    pe = 0.0
    for k in categories:
        p1 = sum(1 for c in cat1 if c == k) / n
        p2 = sum(1 for c in cat2 if c == k) / n
        pe += p1 * p2
    den = 1.0 - pe
    return round((po - pe) / den, 4) if den != 0 else 1.0

def compute_human_llm_agreement(
    human_evals: List[Dict[str, Any]],
    llm_evals: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates 30-50 samples scored by both Human and LLM Judge on 1-5 scale:
    - Pearson Correlation
    - Spearman Correlation
    - Exact Score Agreement %
    - Within ±1 Score Agreement %
    - Cohen's Kappa for rounded score categories
    """
    human_overall = [float(h["overall"]) for h in human_evals]
    llm_overall = [float(l["overall"]) for l in llm_evals]
    
    n = len(human_overall)
    if n == 0:
        return {}

    exact_matches = sum(1 for h, l in zip(human_overall, llm_overall) if round(h) == round(l))
    within_one = sum(1 for h, l in zip(human_overall, llm_overall) if abs(h - l) <= 1.0)

    h_rounded = [int(round(h)) for h in human_overall]
    l_rounded = [int(round(l)) for l in llm_overall]

    pearson = calculate_pearson_correlation(human_overall, llm_overall)
    spearman = calculate_spearman_correlation(human_overall, llm_overall)
    kappa = calculate_cohen_kappa(h_rounded, l_rounded)

    return {
        "sample_size": n,
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "exact_agreement_rate": round(exact_matches / n, 4),
        "within_one_point_agreement_rate": round(within_one / n, 4),
        "cohen_kappa": kappa
    }
