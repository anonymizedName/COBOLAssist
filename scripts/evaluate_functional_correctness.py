# Adapted from: https://github.com/openai/human-eval/blob/master/human_eval/evaluate_functional_correctness.py

import sys

import fire
from evaluation import evaluate_functional_correctness, semantic_fixing
import argparse
from data import HUMAN_EVAL


def entrypoint(model_name,k):
  
    result = eval(f"preds/{model_name}", str(k))

    print(f"{model_name}: {result}")


def eval(
    pred_path: str,
    k: str = "1,10,100",
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k))
    results = evaluate_functional_correctness(pred_path, k, problem_file)#, pred_path+'/program_after_semantic_fixing.csv')
    print(results)
    return results

def entrypoint_fixing():
    model = 'bloopai/mAInframer-7b/'
    # model = 'gpt-4o'
    print(model)
    _semantic_fixing(f"preds/{model}")

def _semantic_fixing(
    pred_path: str,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    results = semantic_fixing(pred_path,problem_file)
    print(results)
    return results

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model', required=True, type=str, help="file_id")
    parser.add_argument('-k', '--k', required=True, type=str, help="file_id")
    args = parser.parse_args()

    return args

def main(args):
    fire.Fire(entrypoint(args.model,args.k))

args = get_args()
sys.exit(main(args))
