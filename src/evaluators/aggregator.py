# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import glob
import copy
import argparse
import jsonlines
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from scipy.stats import bootstrap
from collections import defaultdict
from evaluators.launcher import load_yaml_config

@dataclass
class TraceAnalysisResult:
    """for tracking the results while evaluating a test response"""
    is_failed: bool         # failed when encountering an incorrect trace line
    total_traces: int       # total number of trace lines in the ground truth
    correct_traces: int     # number of consecutive correct trace lines from the start
    target_trace_lines: List[str] = None  # target trace lines around error
    model_trace_lines: List[str] = None   # model trace lines around error
    program: str = None     # program being traced

class TraceAnalyzer:
    """analyze model response against ground truth"""
    
    @staticmethod
    def analyze_trace(item: Dict) -> TraceAnalysisResult:
        """
        compare item['targets'] and item['responses']
        """
        target = item["targets"][0]
        trace_lines = [x.strip() for x in target.split("\n") if x.strip()]
        total_traces = len(trace_lines)
        
        response = TraceAnalyzer._clean_response(item["responses"][0])
        if not response:
            print(f"Model did not provide valid trace for item {item['id']}")
            return TraceAnalysisResult(True, total_traces, 0, [], [], item.get("program", ""))

        response_lines = [x.strip() for x in response.split("\n") if x.strip()]
        response_lines = TraceAnalyzer._pad_response(response_lines, total_traces)
        
        # Find error position and collect surrounding lines
        error_pos = TraceAnalyzer._count_correct_traces(response_lines, trace_lines)
        target_error_lines = []
        model_error_lines = []
        
        if error_pos < total_traces:
            # get lines around the error line
            start_idx = max(0, error_pos - 10)
            target_error_lines = trace_lines[start_idx:error_pos + 1]
            model_error_lines = response_lines[start_idx:error_pos + 1]
        
        return TraceAnalysisResult(
            is_failed=error_pos < total_traces,
            total_traces=total_traces,
            correct_traces=error_pos,
            target_trace_lines=target_error_lines,
            model_trace_lines=model_error_lines,
            program=item.get("program", "")
        )

    @staticmethod
    def _clean_response(response):
        """remove thinking parts, if there is any, and get the trace part by matching the fixed beginning pattern"""
        if "</think>" in response:
            response = response[response.index("</think>"):]
        try:
            return response[response.index("L2,"):]
        except ValueError:
            return ""

    @staticmethod
    def _pad_response(response_lines: List[str], target_length: int) -> List[str]:
        """pad response lines to match target length"""
        if len(response_lines) < target_length - 1:
            response_lines.extend([''] * (target_length - 1 - len(response_lines)))
        return response_lines

    @staticmethod
    def _count_correct_traces(response_lines: List[str], trace_lines: List[str]) -> int:
        """count number of crrct trace lines until the first error occurs"""
        for i, (resp_line, trace_line) in enumerate(zip(response_lines, trace_lines)):
            if trace_line.strip().endswith(","):
                # only compare line numbers for conditional lines
                trace_line = trace_line.strip().split(",")[0]
                resp_line = resp_line.strip().split(",")[0]
            
            if trace_line != resp_line.replace(" ", "").strip():
                return i
        return i + 1

class EvaluationAggregator:
    """aggregate and analyze trace evaluation results"""

    def __init__(self, voter_sizes: List[int] = [5, 15, 31]):
        self.voter_sizes = voter_sizes
        self.analyzer = TraceAnalyzer()

    def _save_error_info(self, data: List[List[Dict]], metrics: List[List[TraceAnalysisResult]], log_paths: List[str]):
        """Save error information for each voter's folder"""
        for voter_idx, log_path in enumerate(log_paths):
            error_file = os.path.join(os.path.dirname(log_path), "errors.jsonl")
            with jsonlines.open(error_file, "w") as f:
                for item_group, metric_group in zip(data, metrics):
                    item = item_group[voter_idx]
                    metric = metric_group[voter_idx]
                    
                    error_info = {
                        "id": item["id"],
                        "tgt_trace_lines": metric.target_trace_lines,
                        "model_trace_lines": metric.model_trace_lines,
                        "program": metric.program
                    }
                    f.write(error_info)

    def evaluate(self, log_paths: List[str]):
        """
        evaluate model responses and aggregate results
        
        log_paths: list of log files, each corresponding to a voter

            1. load data, group voters' responses by item id
            2. analyze traces
            3. generate summary
            4. if there are multiple voters, perform majority voting
        """

        # load data, group voters' responses by item id
        # data: List[List[Dict]]
        data = self._load_data(log_paths)

        # analyze traces
        # metrics: List[List[TraceAnalysisResult]]
        metrics = self._analyze_traces(data)

        # Save error information for each voter
        self._save_error_info(data, metrics, log_paths)

        # generate summary
        # summary: str
        summary = self._generate_summary(metrics)

        # if there are multiple voters, perform majority voting
        if len(log_paths) > 1:
            for voter_size in self.voter_sizes:
                voted_data = [
                    [self._vote_responses(items, voter_size)]
                    for items in data
                ]
                voted_metrics = self._analyze_traces(voted_data)
                summary += f"\n### Majvote @ {voter_size}\n"
                summary += self._generate_summary(voted_metrics)
        log_folder = os.path.dirname(os.path.dirname(log_paths[0]))

        with open(os.path.join(log_folder, "res_eval.summary"), "w") as f:
            f.write(summary)

    def _load_data(self, log_paths: List[str]) -> List[List[Dict]]:
        """load evaluation data from log files and group by id."""
        data_by_id = defaultdict(list)
        for log_path in log_paths:
            with jsonlines.open(log_path, "r") as f:
                for item in list(f):
                    data_by_id[item['id']].append(item)
        
        # verify all items have same number of results
        expected_len = len(log_paths)

        skipped_ids = []
        for _id, items in data_by_id.items():
            if len(items) != expected_len:
                skipped_ids.append(_id)
                print(f"Skipping {_id} because it has {len(items)} voter responses, expected {expected_len}")
        
        for _id in skipped_ids:
            data_by_id.pop(_id)
        
        return data_by_id.values()

    def _analyze_traces(self, data: List[List[Dict]]) -> List[List[TraceAnalysisResult]]:
        """analyze all traces"""
        results = []

        # iterate over items
        for items in data:
            # analyze response from each voter
            results.append([self.analyzer.analyze_trace(item) for item in items])

        return results

    def _generate_summary(self, metrics: List[List[TraceAnalysisResult]]) -> str:
        """generate evaluation summary"""
        total = len(metrics)
        correct = [[not m.is_failed for m in metric] for metric in metrics]
        trace_lens = [np.mean([m.total_traces for m in metric]) for metric in metrics]
        correct_lens = [np.mean([m.correct_traces for m in metric]) for metric in metrics]
        
        summary = ""
        if len(metrics[0]) == 1:
            # print ground truth trace stats
            summary += f"Total num of eval traces: {total}\n"
            summary += f"Ground-Truth Trace steps: {np.round(np.mean(trace_lens), 2)}\n\n"

            # average over all voters
            summary += f"Single Attempt (Whole-Trace Acc.) Avg over {len(metrics[0])} voters: {np.round(np.array(correct).mean(axis=0).mean(), 4)}\n"
            summary += f"Single Attempt (Steps-To-Err.) Avg over {len(metrics[0])} voters: {np.round(np.mean(correct_lens), 2)}\n"
        else:
            # print ground truth trace stats
            summary += f"Total num of eval traces: {total}\n"
            summary += f"Ground-Truth Trace steps: {np.round(np.mean(trace_lens), 2)}\n\n"

            # average over all voters
            summary += f"Single Attempt (Whole-Trace Acc.) Avg over {len(metrics[0])} voters: {np.round(np.array(correct).mean(axis=0).mean(), 4)}\n"
            summary += f"Single Attempt (Steps-To-Err.) Avg over {len(metrics[0])} voters: {np.round(np.mean(correct_lens), 2)}\n"

            # compute pass@k for each voter size
            for voter_size in self.voter_sizes:
                pass_at_k = np.mean([
                    self._compute_pass_at_k(len(c), sum(c), voter_size)
                    for c in correct
                ])
                summary += f"Pass @ {voter_size}: {np.round(pass_at_k, 4)}\n"
            
        return summary + "\n"
        
    def _compute_pass_at_k(self, n: int, c: int, k: int) -> float:
        """compute pass@k metric
        Args:
            n: total number of traces
            c: number of correct traces
            k: number of voters
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    def _vote_responses(self, items: List[Dict], num_voters: int) -> Dict:
        """apply majority voting to responses"""
        responses = [self.analyzer._clean_response(item['responses'][0]) 
                    for item in items[:num_voters]]
        
        # count response frequencies and get majority vote
        response_counts = defaultdict(int)
        for resp in responses:
            response_counts[resp] += 1
        majority_response = max(response_counts.items(), key=lambda x: x[1])[0]
        
        # copy the first item and replace the responses with the majority vote
        result = copy.deepcopy(items[0])
        result["responses"] = [majority_response]
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate model responses and aggregate results")
    parser.add_argument("--config_file", type=str, default="./configs/XXX.yaml")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = load_yaml_config(yaml.safe_load(f))
        config_version = os.path.splitext(os.path.basename(args.config_file))[0]

    log_paths = sorted(glob.glob(f"{config.eval.result_folder}/{config_version}/results/**/logs.jsonl", recursive=True))
    voter_sizes = [int(i) for i in config.eval.num_voters_aggregate if int(i) <= len(log_paths)]
    aggregator = EvaluationAggregator(voter_sizes=voter_sizes)
    aggregator.evaluate(log_paths)
