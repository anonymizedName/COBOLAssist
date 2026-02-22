import os
import subprocess
from dataclasses import dataclass
import pandas as pd

from loguru import logger

def get_completion(file_path):
        
    raw_df = pd.read_csv(file_path)
    comletions = []
    for i, row in raw_df.iterrows():
        compile = 'compile log after fixing {} round'
        code = 'program after fixing {} round'
        final_code = row['generated program']
        if row['compile success after self-debug']:
            for round in range(1, 4):
                compile_tmp = compile.format(round)
                code_tmp =  code.format(round)
                final_code = row[code_tmp]
                if pd.isna(row[compile_tmp]):
                    break
        comletions.append(final_code)

    return comletions


@dataclass
class Model:
    name: str
    temp: float = 0.0
    samples_per_task: int = 1
    tokenizer: str = None
    prefix_token: str = None
    suffix_token: str = None
    middle_token: str = None
    eos_token: str = None


def cmd(cmd: str) -> bool:
    process = subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
        timeout=5,
    )

    # if process.stderr:
    #     logger.warning(f"Err: {process.stderr}")
    #     logger.warning(f"Return code: {process.returncode}")
    return process.returncode == 0, process.stderr


def cleanup_dylib(name: str):
    try:
        os.remove(f"{name}.dylib")
    except FileNotFoundError:
        logger.warning(f"File {name}.dylib not found")


def cleanup_file(name: str):
    try:
        os.remove(f"{name}")
    except FileNotFoundError:
        logger.warning(f"Exe {name} not found")