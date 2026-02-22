# Adapted from: https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py

import itertools
import math
import os
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import tqdm
from loguru import logger
from utils import cleanup_file, cmd
import pandas as pd
from utils import get_completion

from data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from openai import AzureOpenAI
import marko
from marko.block import FencedCode


deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  

endpoint = os.getenv("ENDPOINT_URL", "")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")  
client_gpt2 = AzureOpenAI(
    api_version='2023-05-15',
    azure_endpoint=endpoint,
    api_key=subscription_key,
)




class ParseError(Exception):
    pass


def extract_code_block(src: str) -> List[str]:
    """
    Extract the first code block from markdown source
    """
    markdown = marko.parse(src)

    def search_for_code(element, code_blocks):
        if isinstance(element, FencedCode):
            code_blocks.append(element.children[0].children)
        elif hasattr(element, "children"):
            for child in element.children:
                search_for_code(child, code_blocks)

    code_blocks = []
    search_for_code(markdown, code_blocks)

    # if len(code_blocks) > 1:
    #     logger.warning("Too many code blocks")
    try:
        block = code_blocks[0]
    except:
        block = "Error"
    return block

def find_index_or_last(lst, k):
    try:
        return lst.index(k)
    except ValueError:
        return len(lst) - 1


def parse(result, result_type, true):
    """
    Parse COBOL value according to Python type
    """
    try:
        match result_type:
            case "Bool":
                return parse_bool(result[0])

            case "Int":
                return parse_int(result[0])

            case "Float":
                return parse_float(result[0])

            case "String":
                return parse_string(result[0])

            case {"List": "Int"}:
                parsed_result = [parse_int(x) for x in result]
                return parsed_result[: len(true)]

            case {"List": "Float"}:
                parsed_result = [parse_float(x) for x in result]
                return parsed_result[: len(true)]

            case {"List": "String"}:
                parsed_result = [parse_string(x) for x in result]
                return parsed_result[: len(true)]

            case _:
                raise ParseError("Invalid result type: ", result_type)
    except Exception as e:
        raise ParseError(f"Result {result} of type {result_type} failed with: {e}")


def parse_bool(res: str) -> bool:
    if res.strip() == "1":
        return True
    return False


def parse_int(res: str) -> int:
    res = res.strip()
    if res.startswith("p") or res.startswith("y"):
        return -int(res[1:])
    return int(res)


def parse_float(res: str) -> float:
    res = res.strip()
    if res.startswith("p") or res.startswith("y"):
        res = res[1:]
        return -float(res)
    return float(res)


def parse_string(res: str) -> str:
    return res.strip()


def is_equal(result_type, result, true):
    match result_type:
        case "Float":
            return math.isclose(result, true, abs_tol=0.001)
        case {"List": "Float"}:
            return all(math.isclose(r, t, abs_tol=0.001) for r, t in zip(result, true))
        case _:
            return result == true


def exec(name, path, call_path) -> bool:
    cmp, e = cmd(f"cobc -w -fformat=variable -x {call_path} {path}")
    if not cmp:
        logger.warning(f"Compile error for {path}")
        return False

    # WARNING
    # This program exists to execute untrusted model-generated code. Although
    # it is highly unlikely that model-generated code will do something overtly
    # malicious in response to this test suite, model-generated code may act
    # destructively due to a lack of model capability or alignment.
    # Users are strongly encouraged to sandbox this evaluation suite so that it
    # does not perform destructive actions on their host or network.
    # Once you have read this disclaimer and taken appropriate precautions,
    # uncomment the following lines and proceed at your own risk:

    cmp, error = cmd(f"./call_{name}")
    if not cmp:
        logger.warning(f"Runtime error for {path}")
        return False

    return True


def check_correctness(problem: Dict, completion: str, base_path: str) -> Dict:
    """
    Check the correctness of a single completion
    """
    # completion = completion.replace("* ", "*> ")
    # completion = completion.replace("```cobol", "")
    name, tests = problem["entry_point"], problem["tests"]
    path, call_path = (
        os.path.join(base_path, f"solutions/{name}.cbl"), #generated code 
        os.path.join(base_path, f"callers/call_{name}.cbl"),
    )
    result_path = f"{name.upper().replace('_', '-')}.TXT"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(os.path.dirname(call_path), exist_ok=True)

    try:
        with open(path, "w") as f:
            f.write(completion)
    except:
        with open(path, "w") as f:
            f.write("completion")

    passed, trues, results, compiled = [], [], [], []
    for test in tests:
        true = eval(test["result"]["value"])
        if isinstance(true, tuple):  # convert tuples to list
            true = list(true)

        trues.append(true)
        passed.append(False)
        results.append(None)
        compiled.append(False)

        with open(call_path, "w") as f:
            f.write(test["test"]) # testcase

        try:
            if exec(name, path, call_path):
                compiled[-1] = True
                # print(completion)

                with open(result_path) as f:
                    result = f.readlines()

                if result:
                    type_ = test["result"]["type_"]
                    parsed_result = parse(result, type_, true)
                    passed[-1] = is_equal(type_, parsed_result, true)
                    results[-1] = parsed_result
        except Exception as e:
            logger.error(f"Eval {name} failed with: {e}")
        finally:
            cleanup_file(f"call_{name}")
            cleanup_file(result_path)
    if compiled[-1] == False:
        print(completion)
    return {
        "all_passed": all(passed),
        "passed": passed,
        "results": results,
        "trues": trues,
        "compiled": compiled,
    }


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def get_clean_response(str):
    if '<|EOT|>' in str:
        str = str[:str.index('<|EOT|>')]
    if '```' in str:
        str = str[:str.index('```')]
    if ' The provided' in str:
        str = str[:str.index(' The provided')]
    if 'The provided' in str:
        str = str[:str.index('The provided')]
    if 'This COBOL' in str:
        str = str[:str.index('This COBOL')]
    # str = str.replace("*", "*>")
    if str[-1] == '.':
        str += '\n'
    return str

def read_task(file):
    return [task["task_id"] for task in stream_jsonl(file)]

def get_clean_response_for_Xmainframe(code):
    tmp = """WORKING-STORAGE SECTION.
       WORKING-STORAGE SECTION."""
    code = code.replace(tmp, 'WORKING-STORAGE SECTION.')
    try: 
        working_index  = code.index("WORKING-STORAGE SECTION.")
        try:
            procedure_index = code.index("PROCEDURE DIVISION USING")
        except:
            procedure_index = code.index("PROCEDURE DIVISION.")
        linkage_index = code.index("LINKAGE SECTION.")
        if working_index > linkage_index:
            working_division = code[working_index:procedure_index]
            prefix = code[:linkage_index]
            procedure = code[procedure_index:]
            likage_section = code[linkage_index:working_index]

            code = prefix + working_division + likage_section + procedure
    except:
        pass

    return code
def evaluate_functional_correctness(
    base_path: str,
    k: List[int] = [1, 10, 100],
    problem_file: str = HUMAN_EVAL,
    path = ''
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    problems = read_problems(problem_file)
    task_ids = read_task(problem_file)

    sample_file = base_path + ".csv"
    df = pd.read_csv(sample_file)
    calls_path = os.path.join(base_path, "callers")
    os.makedirs(calls_path, exist_ok=True)

    n_samples = 0
    results = defaultdict(list)

    logger.info("Reading samples...")


    for id, sample in df.iterrows():
        num_pass = -1
        true_correct = None
        for time in range(k[0]):
            id_, task_id, completion = (
                0,
                task_ids[id],
                sample[f"output_{time}"],
                # completions[id]
            )
       
            correct = check_correctness(problems[task_id], completion, base_path)
            if sum(correct['passed']) > num_pass:
                num_pass = sum(correct['passed'])
                true_correct = correct

        n_samples += 1
        results[task_id].append((id_, true_correct))
       


    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["all_passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    k = [1]
    ks = k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }

    total = 0
    passed = 0
    compiled = 0
    pass_rate = 0

    for result in results.values():
        for r in result:
            total += len(r[1]["passed"])
            passed += sum(r[1]["passed"])
            compiled += sum(r[1]["compiled"])
            pass_rate += (len(r[1]["passed"]) == sum(r[1]["passed"]))

    logger.info(f"Total tests: {total}, Passed: {passed}, Compiled: {compiled}, Pass_rate: {pass_rate}")

    # Finally, save the results in one file:
    def combine_results():
        for i, sample in enumerate(stream_jsonl(sample_file)):
            task_id = task_ids[i]
            result = results[task_id].pop(0)
            sample["trues"] = result[1]["trues"]
            sample["passed"] = result[1]["passed"]
            sample["results"] = result[1]["results"]
            sample["compiled"] = result[1]["compiled"]
            sample["all_passed"] = result[1]["all_passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    logger.info(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k

def get_response_gpt(prompt, client, model, temperature=0.2):
    messages = [ {"role": "system", "content": "You are a COBOL software developer."} ]
    messages.append(
        {"role": "user", "content": prompt},
    )
    chat = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    reply = chat.choices[0].message.content
    return reply


def get_semantic_prompt(code, unit_test, expected_output, actual_output):
    prompt = """You are a COBOL software developer. Your task is to debug and fix the given COBOL code to ensure it produces the correct output when executing the provided unit test.
Below, you are given the COBOL code, the unit test, the expected output for the test, and the actual output from running the COBOL code.
Analyze the discrepancies between the expected output and the actual output.
Identify and fix any logical, semantic, or runtime errors in the COBOL code.
Modify the COBOL code so that it correctly produces the expected output when running the unit test.
Ensure that the fixed code remains logically sound, syntactically correct, and adheres to COBOL best practices.
COBOL Code: {code}
Unit Test: {unit_test}
Expected Output: {expected_output}
Actual Output: {actual_output}
""".format(code=code, unit_test=unit_test, expected_output=expected_output, actual_output=actual_output)
    
    return prompt


def semantic_fixing(
    base_path: str,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    problems = read_problems(problem_file)

    sample_file = os.path.join(base_path, "samples.jsonl")
    solutions_path = os.path.join(base_path, "solutions")
    calls_path = os.path.join(base_path, "callers")
    os.makedirs(solutions_path, exist_ok=True)
    os.makedirs(calls_path, exist_ok=True)

    logger.info("Reading samples...")
    completions = get_completion(os.path.join(base_path, "program_after_fixing.csv"))

    df = pd.DataFrame(columns=['generated program'])
    program_name = "program after fixing {} round"
    for i in range(1,4):
        p = program_name.format(i)
        df[p] = ''

    df['generated program'] = completions
    df['fixable'] = False

    for id, sample in enumerate(list(stream_jsonl(sample_file))):
        _, task_id, completion = (
            sample["sample_id"],
            sample["task_id"],
            completions[id]
        )

        correct = check_correctness(problems[task_id], completion, base_path)
        
        code = ''
        for i, pa in enumerate(correct['passed']):
            if not pa:
                unit_test = problems[task_id]["tests"][i]['test']
                expected_output = correct['trues'][i]
                output = correct['results'][i]
                code = completions[id]
                test = problems[task_id]["tests"][i]
                true = eval(test["result"]["value"])
                if isinstance(true, tuple):  # convert tuples to list
                    true = list(true)
                break

        if code != '':
            self_debug_round = 1
            tmp_path, call_path = (
                os.path.join(base_path, f"solutions/{problems[task_id]['entry_point']}.cbl"), #generated code 
                os.path.join(base_path, f"callers/call_{problems[task_id]['entry_point']}.cbl"),
            ) 
            result_path = f"{problems[task_id]['entry_point'].upper().replace('_', '-')}.TXT"
            with open(call_path, "w") as f:
                f.write(unit_test)

            PASS = False
            while self_debug_round < 4:
                prompt = get_semantic_prompt(code, unit_test, expected_output, output)
                response = get_response_gpt(prompt,client_gpt2, deployment)

                code = extract_code_block(response)
                try:
                    with open(tmp_path, "w") as f:
                        f.write(completion)
                except:
                    with open(tmp_path, "w") as f:
                        f.write("completion")
                    print(completion)
                
                df.at[id, "program after fixing {} round".format(str(self_debug_round))] = code
                
                try:
                    if exec(problems[task_id]['entry_point'], tmp_path, call_path):
                        with open(result_path) as f:
                            result = f.readlines()

                        if result:
                            type_ = test["result"]["type_"]
                            parsed_result = parse(result, type_, true)
                            PASS = is_equal(type_, parsed_result, true)
                except Exception as e:
                    logger.error(f"Eval {problems[task_id]['entry_point']} failed with: {e}")
                finally:
                    cleanup_file(f"call_{problems[task_id]['entry_point']}")
                    cleanup_file(result_path)
                
                if PASS:
                    df.at[id, 'fixable'] = True
                    break
                self_debug_round += 1

    df.to_csv(os.path.join(base_path, "program_after_semantic_fixing.csv"), index=False)