#!/usr/bin/env python3
"""Evaluate existing benchmark JSONL responses without calling any model endpoint.

This script decouples the evaluation path from benchmark-vanilla.py.
It reads an existing JSONL file, evaluates each record's response against the
provided ground truth, writes the computed performance to a new JSONL file, and
never overwrites the input file or an existing output file.

Rules:
1. If a record has no response field or response is null, set performance to 0.
2. If a record has a response, reuse the original evaluation logic to compute
   performance from response, ground_truth, metric, and choices.
3. Always write to a new output file.

Usage:
    python benchmark-vani-evalonly.py --input benchmark_output.jsonl
    python benchmark-vani-evalonly.py --input benchmark_output.jsonl --output rescored.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def _evaluate_multiple_choice(
    response: str, ground_truth: str, choices: Optional[str]
) -> float:
    """Evaluate multiple choice questions by extracting the answer letter."""
    response_text = response.strip()
    truth_upper = ground_truth.upper().strip()

    if len(truth_upper) == 1 and truth_upper in "ABCDEFGHIJ":
        parenthesis_pattern = re.findall(r"\(\s*([a-zA-Z])\s*\)", response_text)
        if parenthesis_pattern:
            found_letter = parenthesis_pattern[-1].upper()
            return 1.0 if found_letter == truth_upper else 0.0

        patterns = [
            r"(?:answer(?:\s*is)?:?\s*)([A-J])\b",
            r"(?:it['\u2019]?s|is)\s+([A-J])\b",
            r"['\u2019]s\s+([A-J])\b",
            r"\b([A-J])\s+(?:because|since|as)",
            r"(?:think|believe|choose)\s+([A-J])\b",
            r"\b([A-J])\s*[.)\]:]",
            r"^([A-J])[.)\]:\s]*$",
            r"\b([A-J])$",
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                found_letter = match.group(1).upper()
                if found_letter == "I" and not re.match(
                    r"^I[.)\]:\s]*$", response_text.strip(), re.IGNORECASE
                ):
                    continue
                return 1.0 if found_letter == truth_upper else 0.0

        if truth_upper != "I" and re.search(
            r"\b" + truth_upper + r"\b", response_text.upper()
        ):
            return 0.8
        if truth_upper == "I" and re.search(
            r"(?:answer|choice|option)[:\s]+I\b", response_text, re.IGNORECASE
        ):
            return 0.8

    return 0.0


def _evaluate_gsm8k(response: str, ground_truth: str) -> float:
    """Evaluate GSM8K math problems."""
    if "####" in ground_truth:
        ground_truth_processed = ground_truth.split("####")[-1]
    else:
        ground_truth_processed = ground_truth

    ground_truth_processed = (
        ground_truth_processed.replace(",", "")
        .replace("$", "")
        .replace(".", "")
        .strip()
    )

    numbers = re.findall(r"(\-?[0-9\.\,]+)", response)
    if not numbers:
        return 0.0

    final_answer = None
    for answer in reversed(numbers):
        if answer not in {"", "."}:
            final_answer = answer
            break

    if final_answer is None:
        return 0.0

    final_answer = final_answer.replace(",", "").replace("$", "").replace(".", "").strip()
    return 1.0 if final_answer == ground_truth_processed else 0.0


def _strip_latex_string(string: str) -> str:
    """Normalize LaTeX string for comparison."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if string and string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = string.replace(" ", "")
    return string.strip()


def _last_boxed_string(text: str) -> Optional[str]:
    """Extract the last \\boxed{} content from text."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def _remove_boxed(text: str) -> str:
    """Remove \\boxed{} wrapper and return its content."""
    if "\\boxed{" in text:
        start = text.find("\\boxed{") + len("\\boxed{")
        depth = 1
        end = start
        while end < len(text) and depth > 0:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        return text[start : end - 1]
    if "\\boxed " in text:
        return text.split("\\boxed ")[-1].split()[0]
    return text


def _evaluate_math(response: str, ground_truth: str) -> float:
    """Evaluate MATH problems by extracting boxed answers."""
    gt_boxed = _last_boxed_string(ground_truth)
    if gt_boxed:
        ground_truth_processed = _remove_boxed(gt_boxed)
    else:
        ground_truth_processed = ground_truth.strip()

    try:
        response_boxed = _last_boxed_string(response)
        if response_boxed:
            response_answer = _remove_boxed(response_boxed)
            if _strip_latex_string(response_answer) == _strip_latex_string(
                ground_truth_processed
            ):
                return 1.0
    except Exception:
        pass

    gt_normalized = _strip_latex_string(ground_truth_processed)
    response_normalized = _strip_latex_string(response)
    if gt_normalized and gt_normalized in response_normalized:
        return 0.8

    try:
        gt_nums = re.findall(r"-?\d+\.?\d*", ground_truth_processed)
        resp_nums = re.findall(r"-?\d+\.?\d*", response)
        if gt_nums and resp_nums and gt_nums[-1] in resp_nums:
            return 0.7
    except Exception:
        pass

    return 0.0


def _evaluate_f1(response: str, ground_truth: str) -> float:
    """Calculate F1 score based on word overlap."""
    import string

    def clean_words(text: str) -> set[str]:
        text = text.lower()
        for punctuation in string.punctuation:
            text = text.replace(punctuation, " ")
        return set(text.split())

    response_words = clean_words(response)
    truth_words = clean_words(ground_truth)
    if not truth_words:
        return 0.0

    if len(truth_words) <= 2:
        truth_text = ground_truth.lower()
        for punctuation in string.punctuation:
            truth_text = truth_text.replace(punctuation, "")
        if truth_text.strip() in response.lower():
            return 1.0

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "in",
        "to",
        "and",
        "or",
    }
    response_content = response_words - stopwords
    truth_content = truth_words - stopwords
    if not truth_content:
        truth_content = truth_words
    if not response_content:
        response_content = response_words

    overlap = response_content & truth_content
    if not overlap:
        return 0.0

    precision = len(overlap) / len(response_content) if response_content else 0.0
    recall = len(overlap) / len(truth_content)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _evaluate_code(response: str, ground_truth: str, timeout: int = 5) -> float:
    """Evaluate code by executing assertion lists when available."""
    import signal

    code_patterns = [
        r"```python\n(.*?)```",
        r"```\n(.*?)```",
        r"def\s+\w+\s*\([^)]*\):.*?(?=\n\n|\Z)",
    ]

    code = response
    for pattern in code_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1) if match.lastindex else match.group(0)
            break

    try:
        if ground_truth.startswith("[") and "assert" in ground_truth:
            assertions = eval(ground_truth)
            if isinstance(assertions, list):
                passed = 0
                total = len(assertions)

                def timeout_handler(signum, frame):
                    raise TimeoutError("Code execution timed out")

                alarm_supported = hasattr(signal, "SIGALRM")
                for assertion in assertions:
                    try:
                        if alarm_supported:
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(timeout)

                        local_vars: Dict[str, Any] = {}
                        exec(code, {}, local_vars)
                        exec(assertion, local_vars)
                        passed += 1
                    except (AssertionError, TimeoutError):
                        pass
                    except Exception:
                        pass
                    finally:
                        if alarm_supported:
                            signal.alarm(0)

                return passed / total if total > 0 else 0.0
    except Exception:
        pass

    func_match = re.search(r"def\s+(\w+)", response)
    if func_match and func_match.group(1) in ground_truth.lower():
        return 0.5
    return 0.3


def _evaluate_commongen(response: str, ground_truth: str) -> float:
    """Evaluate CommonGen by checking target word coverage."""
    required_words = set(word.strip().lower() for word in ground_truth.split(","))
    response_lower = response.lower()
    found = sum(1 for word in required_words if word in response_lower)
    return found / len(required_words) if required_words else 0.0


def _normalize_answer(text: str) -> str:
    """Normalize text for comparison."""
    import string

    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def _evaluate_cem(response: str, ground_truth: str) -> float:
    """Conditional exact match evaluation."""
    norm_response = _normalize_answer(response)
    norm_gt = _normalize_answer(ground_truth)
    if norm_response == norm_gt or norm_gt in norm_response:
        return 1.0
    return 0.0


def evaluate_response(
    response: str,
    ground_truth: Optional[str],
    metric: Optional[str] = None,
    choices: Optional[str] = None,
) -> float:
    """Evaluate a response using the same metric-specific logic as benchmark-vanilla.py."""
    if ground_truth is None:
        return 0.5

    response_lower = response.lower().strip()
    truth_lower = ground_truth.lower().strip()
    if response_lower == truth_lower:
        return 1.0

    if metric == "em_mc" or choices:
        return _evaluate_multiple_choice(response, ground_truth, choices)
    if metric == "GSM8K":
        return _evaluate_gsm8k(response, ground_truth)
    if metric == "MATH":
        return _evaluate_math(response, ground_truth)
    if metric == "f1_score":
        return _evaluate_f1(response, ground_truth)
    if metric == "code_eval":
        return _evaluate_code(response, ground_truth)
    if metric == "commongen_coverage":
        return _evaluate_commongen(response, ground_truth)
    return _evaluate_cem(response, ground_truth)


def default_output_path(input_path: Path) -> Path:
    """Build a non-destructive default output path next to the input file."""
    return input_path.with_name(f"{input_path.stem}.evalonly{input_path.suffix}")


def validate_paths(input_path: Path, output_path: Path) -> None:
    """Reject destructive path combinations before any work starts."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output file must be different from input file")
    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")


def evaluate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the input record with its performance field refreshed."""
    result = dict(record)
    response = result.get("response")
    if response is None:
        result["performance"] = 0.0
        return result

    result["performance"] = evaluate_response(
        str(response),
        result.get("ground_truth"),
        result.get("metric"),
        result.get("choices"),
    )
    return result


def process_file(input_path: Path, output_path: Path, show_progress: bool = True) -> int:
    """Process a JSONL file line by line and write rescored output to a new file."""
    validate_paths(input_path, output_path)

    total_written = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "x", encoding="utf-8"
    ) as dst:
        iterator = src
        if show_progress:
            iterator = tqdm(src, desc="Evaluating")

        for line_num, line in enumerate(iterator, 1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_path} at line {line_num}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {input_path} at line {line_num}"
                )

            evaluated = evaluate_record(record)
            dst.write(json.dumps(evaluated, ensure_ascii=False) + "\n")
            total_written += 1

    return total_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate existing benchmark JSONL responses without re-running inference"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL file containing response fields",
    )
    parser.add_argument(
        "--output",
        help="Path to output JSONL file. Must not already exist. Default: <input>.evalonly.jsonl",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    try:
        written = process_file(
            input_path=input_path,
            output_path=output_path,
            show_progress=not args.no_progress,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {written} evaluated records to {output_path}")


if __name__ == "__main__":
    main()