

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    
    solution_str = solution_str.split('\n')[-1]

    
    final_answer = remove_boxed(last_boxed_only_string(solution_str))
    return final_answer

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


if __name__ == "__main__":
    print(remove_boxed(last_boxed_only_string("\\boxed{123}")))
    print(remove_boxed(last_boxed_only_string("\\fbox{123}")))
    print(remove_boxed("123"))
    print(remove_boxed(last_boxed_only_string("\\boxed{123}123")))
    print(remove_boxed(last_boxed_only_string("\\fbox{123}123")))