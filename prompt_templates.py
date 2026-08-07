# setting of <《language-ranker》 (nips25)
# mathematical reasoning, code-generation, tool-using

SYSTEM_MATH = "You are a math expert."

USER_MATH = """Please solve the given math problem step by step and present the answer in the following format: "\\boxed{{X}}", where X is the answer.\n\nQuestion:\n{question}"""

SYSTEM_CODE = "You are an expert Python programmer."

USER_CODE = """Write a Python function based on the following instructions and test example. Please ensure that the function is clearly marked with a start and end so I can easily extract it from your output.\n\nInstructions:\n{question}\n\nTest Example:\n{test_list[0]}\n\nPlease provide your code with clear start and end markers, like so:\n\n# START OF CODE\ndef {function_name}(input):\n    # Function code here\n    return result\n# END OF CODE"""


SYSTEM_TOOL = """You are a function-calling assistant. Your role is to complete tasks solely through correct function calls, without generating any additional text. For each task, directly output the function call(s) required to complete it. If the task involves multiple steps, you may issue multiple function calls sequentially. 
Each function call must be formatted as a JSON object. For example: [{\"name\": \"function_A\", \"arguments\": {\"param1\": \"value1\", \"param2\": \"value2\"}}, {\"name\": \"function_B\", \"arguments\": {\"param1\": \"value1\", \"param2\": \"value2\"}}]

The following are the available functions:
{function_list}

Now, use the appropriate function(s) to complete the given task.
"""

USER_TOOL = """{query}
Please directly output the function call(s) to solve the task without any other text."""



# setting of ours
############         math               #####################

# actor4math
system_prompt4math = 'You are a math expert.'
user_prompt4math = (
    'Please solve the given math problem step by step and present the answer in the following format: "\\boxed{{X}}", where X is the answer.'
    '\n\nQuestion:\n{question}'
)

# judge4math
system_prompt4judge = (
    'You are a careful math judge. Verify whether a student solution matches the correct answer. '
    'Focus on mathematical equivalence of the final answer; ignore minor formatting issues or missing units.'
)
user_prompt4judge = (
    'I have a full chain-of-thought solution to a math problem that needs verification.\n\n'
    'Correct answer: {answer}\n\n'
    'Solution:\n{res}\n\n'
    'Please briefly check:\n'
    '1) What final answer does the solution claim?\n'
    '2) Is that answer mathematically equivalent to the correct answer?\n'
    '3) If the final answer is missing, incomplete, or clearly wrong, mark Incorrect.\n\n'
    'You may briefly reason step by step first (keep it short and simple). After your reasoning, output the final judgment '
    'on its own last line in exactly one of these two forms:\n'
    'Verdict: Correct\n'
    'or\n'
    'Verdict: Incorrect\n'
    'Do not put any other text after the Verdict line.'
)
#