TASK_FREQUENCY = {
    "summary": 1,
    "unfilter": 4,
    "generated_qna": 2,
    "generated_logic_qna": 1,
    "generated_tool_selection": 0,
    "tool_call": 6,
    "tool_gen": 4,
    "conversation": 2,
}

TASK_WEIGHTS = {
    "basic_qna": 0.0,
    "summary": 0.04,
    "unfilter": 0.09,
    "generated_qna": 0.03,
    "generated_logic_qna": 0.03,
    "generated_tool_selection": 0.00,
    "tool_call": 0.1,
    "tool_call_dataset": 0.025,
    "tool_gen": 0.1,
    "tool_gen_dataset": 0.025,
    "conversation": 0.03,
}
