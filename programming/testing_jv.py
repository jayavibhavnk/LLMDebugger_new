from openai import OpenAI
import json
import re
import ast
import random
import pandas as pd
import unittest
from io import StringIO
from programming.generators import PyGenerator, model_factory
from programming.executors import PyExecutor
from programming.utils import get_func_header, insert_comment, IMPORT_HEADER


def generate_seed_code(prompt, test, temperature=0.0, model=MODEL_NAME):
    client = OpenAI(api_key=OPENAI_KEY)
    user_prompt = f"Complete the following task in Python. Here is a unit test:\n{test.strip()}\n\n{prompt.rstrip()}"
    messages = [
        {"role": "system", "content": "You are an expert programming assistant. You will return the code only along with any import statements or additonal requirements, the code should be returned directly no bacticks, no extra characters, import statements are allowed"},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=1
    )
    return response.choices[0].message.content, messages

def read_first_jsonl(filename):
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return None

def run_unittest(code, test_code):
    local_globals = {}
    try:
        exec(code, local_globals)
        exec(test_code, local_globals)
    except Exception as e:
        return False, None, f"Execution Error: {str(e)}"
    if "TestCases" not in local_globals:
        return False, None, "No TestCases found in executed code."
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(local_globals.get("TestCases"))
    tests = []
    for t in suite:
        tests.append(t)
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    test_output = stream.getvalue()
    test_results = []
    for test in tests:
        test_name = test._testMethodName
        status = "Pass"
        for fail in result.failures:
            if fail[0]._testMethodName == test_name:
                status = "Fail"
        for error in result.errors:
            if error[0]._testMethodName == test_name:
                status = "Error"
        test_results.append((test_name, status))
    df = pd.DataFrame(test_results, columns=["Test", "Status"])
    return result.wasSuccessful(), df, test_output

def test_code_unittest(code, test_code):
    passing, df, output = run_unittest(code, test_code)
    return df, code, output

def debug_code_unittest(openai_key, model, task_prompt, code, test_code, entry_point):
    passing, df, test_output = run_unittest(code, test_code)
    if passing:
        return "Tests already pass", code, code, df, test_output
    failing_tests_df = df[df["Status"] != "Pass"]
    failing_test = failing_tests_df.iloc[0]["Test"]
    if "# Real Execution Output:" not in failing_test:
        failing_test = f"{failing_test} # Real Execution Output: Unknown"
    func_header = get_func_header(code, entry_point)
    prompt_debug = insert_comment(func_header, task_prompt, entry_point)
    code_with_comment = insert_comment(code, task_prompt, entry_point)
    generator = PyGenerator()
    model_obj = model_factory(model, key=openai_key)
    dataset_type = "Unittest"
    raw_messages = generator.ldb_debug(
        prompt_debug,
        code_with_comment,
        failing_test,
        entry_point,
        model_obj,
        "",
        dataset_type,
        "block"
    )
    messages_list = []
    for m in raw_messages:
        if hasattr(m, "content"):
            messages_list.append(m.content.strip())
        else:
            messages_list.append(str(m))
    debug_log = "======== Prompt ========\n"
    for i, m in enumerate(messages_list):
        if i == 0:
            debug_log += "----- System -----\n" + m + "\n"
        elif i == len(messages_list) - 1:
            debug_log += "\n======== Response ========\n" + m
        else:
            if i % 2 == 1:
                debug_log += "----- User -----\n" + m + "\n"
            else:
                debug_log += "----- Assistant -----\n" + m + "\n"
    messages_str = "\n".join(messages_list)
    fixed_code, new_messages = generator.ldb_generate(
        func_sig=task_prompt,
        model=model_obj,
        prev_func_impl=code_with_comment,
        messages=messages_str,
        failed_tests=failing_test,
        dataset_type=dataset_type,
    )
    passing_new, df_new, test_output_new = run_unittest(fixed_code, test_code)
    return debug_log, code, fixed_code, df_new, test_output_new

def log_iteration(log_filename, log_entry):
    with open(log_filename, "a") as f:
        json.dump(log_entry, f)
        f.write("\n")

def run_pipeline(task, openai_key, model_name, max_iterations=3, log_filename="pipeline_log.jsonl"):
    prompt = task.get("prompt", "")
    test_code = task.get("test", "")
    seed_code, seed_messages = generate_seed_code(prompt, test_code)
    code = seed_code
    entry_point = task.get("entry_point", "").strip()
    iteration = 1
    overall_log = []
    while iteration <= max_iterations:
        try:
            print(f"\n--- Iteration {iteration}: Testing generated code ---")
            df, tested_code, test_output = test_code_unittest(code, test_code)
            passing, _, _ = run_unittest(code, test_code)
            try:
                log_entry = {
                    "task_id": task.get("task_id", ""),
                    "iteration": iteration,
                    "phase": "test",
                    "code": code,
                    "test_output": test_output,
                    "test_results": df.to_dict(orient="records"),
                    "passing": passing
                }
            except:
                log_entry = {
                    "task_id": task.get("task_id", ""),
                    "iteration": iteration,
                    "phase": "test",
                    "code": code,
                    "test_output": test_output,
                    "test_results": {},
                    "passing": False
                }
            log_iteration(log_filename, log_entry)
            overall_log.append(log_entry)
            print("Test Results:")
            print(df)
            if passing:
                print("All tests pass!")
                return code, df, overall_log
            print(f"\n--- Iteration {iteration}: Debugging generated code ---")
            debug_log, original_code, fixed_code, df_new, test_output_new = debug_code_unittest(
                openai_key, model_name, prompt, code, test_code, entry_point
            )
            try:
                log_entry_debug = {
                    "task_id": task.get("task_id", ""),
                    "iteration": iteration,
                    "phase": "debug",
                    "original_code": original_code,
                    "debug_log": debug_log,
                    "fixed_code": fixed_code,
                    "test_output": test_output_new,
                    "test_results": df_new.to_dict(orient="records")
                }
            except:
                log_entry_debug = {
                    "task_id": task.get("task_id", ""),
                    "iteration": iteration,
                    "phase": "debug",
                    "original_code": original_code,
                    "debug_log": debug_log,
                    "fixed_code": fixed_code,
                    "test_output": test_output_new,
                    "test_results": {}
                }
            log_iteration(log_filename, log_entry_debug)
            overall_log.append(log_entry_debug)
            code = fixed_code
            iteration += 1
        except:
            iteration += 1
            continue
        print("\nMaximum iterations reached. Final code may not pass all tests.")
    df_new = pd.DataFrame()
    return code, df_new, overall_log

if __name__ == '__main__':
    input_filename = "/Users/jvnk/Documents/codedebug/LLMDebugger_new/input_data/bigcode/dataset/probs_oldn.jsonl"
    log_filename = "pipeline_log.jsonl"
    tasks = []
    with open(input_filename, "r") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line.strip()))
    for task in tasks:
        print(f"\n### Processing Task: {task.get('task_id', 'unknown')} ###")
        final_code, final_df, logs = run_pipeline(task, OPENAI_KEY, MODEL_NAME, max_iterations=3, log_filename=log_filename)
        print("\n=== Final Code ===")
        print(final_code)
        print("\n=== Final Test Results ===")
        print(final_df)
