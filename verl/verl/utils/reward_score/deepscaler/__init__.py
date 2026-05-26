"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union


from verl.utils.reward_score.deepscaler.reward_type import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from verl.utils.reward_score.deepscaler.math_utils import extract_answer, grade_answer_sympy, grade_answer_mathd
import random

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

PRINT_PROB = 0.0005

# FORMAT_REWARD = 0.0 #TODO: don't use format for not
# CORRECT_REWARD = 1.0
# INVALID_REWARD = 0.0

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput, correct_reward, format_reward) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)

        do_print = random.random() < PRINT_PROB
        
        problem = input.problem
        model_response = input.model_response

        if do_print:
            print('-' * 50)
            print(f"Model response: {model_response}")
        
        # Extract solution.
        if "<|im_start|>assistant" in model_response:
            model_solution = model_response.split("<|im_start|>assistant")[1]
        else:
            model_solution = model_response


        if THOUGHT_DELIMITER_START in model_solution and THOUGHT_DELIMITER_END in model_solution:
            model_solution = model_solution.split(THOUGHT_DELIMITER_END)[-1]
        # else:
        #     if do_print:
        #         print('Invalid model response format')
        #     return RewardOutput(reward=INVALID_REWARD, is_correct=False)
        model_solution = model_solution[-500:]
        
        model_answer = extract_answer(model_solution)
        if do_print:
            print(f"Model answer: {model_answer}")
        if model_answer is None or model_answer == "":
            if do_print:
                print('No answer')
            return RewardOutput(reward=0.0, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            if do_print:
                print('No ground truth found')
            return RewardOutput(reward=format_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=format_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                if do_print:
                    print('Correct answer')
                return RewardOutput(reward=correct_reward, is_correct=True)
        if do_print:
            print('Wrong answer')
        return RewardOutput(reward=format_reward, is_correct=False)


def get_deepscaler_reward_fn(correct_reward, format_reward):

    def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
        reward_config = RewardConfig()
        reward_config.use_math_orm = enable_llm
        reward_fn = RewardMathFn(reward_config)
        reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}), correct_reward, format_reward)
        return reward_response.reward
    
    return deepscaler_reward_fn

