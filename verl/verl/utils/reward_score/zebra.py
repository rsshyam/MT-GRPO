import re
import random
from verl.utils.reward_score.utils import remove_boxed, last_boxed_only_string, extract_solution

PRINT_EVERY = 4096



def evaluate_equation(answer, target):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    if answer == None:
        return False
    if answer.lower().strip() != target.lower().strip():
        return False
    else:
        return True



def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, PRINT_EVERY) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0
    
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation, target)

        if result:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {equation}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score 
    
def get_zebra_compute_score(correct_score, format_score):
    return lambda solution_str, ground_truth: compute_score(solution_str, ground_truth, method='strict', format_score=format_score, score=correct_score)
