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
import json
import yaml
import random
import pickle
import argparse
import tiktoken
import datetime
import jsonlines
import numpy as np
from tqdm import tqdm
from typing import List
from tqdm.contrib.concurrent import thread_map
from transformers import AutoTokenizer
from evaluators.client import Client
from evaluators.launcher import load_yaml_config


class Evaluator:

    def __init__(
        self, 
        client: Client = None, 
        tokenizer: AutoTokenizer = None,
        num_workers: int = 20,
        random_seed: int = 42,
        log_path: str = "", 
        data_path: str = "",
        data_file: str = "",
        data_size: int = 500,
        num_fewshot: int = 2,
        fewshot_file: str = "",
        fewshot_pool_size: int = 64,
        fewshot_type: str = "input_trace",
        prompt_file: str = "", 
        stop: str = "",
    ):  

        self.client = client
        self.model_name = "none" if client is None else self.client.model_name.split("/")[-1].lower()
        self.tokenizer = tokenizer
        self.num_workers = num_workers  # batch size
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.evaluation_data = self.load_evaluation_data(data_path, data_file, data_size)
        self.num_fewshot = num_fewshot
        self.fewshot_type = fewshot_type 
        self.fewshot_examples = self.load_fewshot_examples(data_path, fewshot_file, fewshot_pool_size)
        
        self.prompt_template = self.load_prompt_template(prompt_file)
        self.stop_token = stop
        self.output_dir = log_path
        os.system(f"mkdir -p {self.output_dir}")
        
    def load_evaluation_data(self, data_dir, data_filename, max_samples):
        """
            Load data
                1.load data split indices 
                2.filter data.jsonl
        """
        with open(os.path.join(data_dir, data_filename), "rb") as f:
            sample_indices = set(pickle.load(f))
            
        evaluation_samples = []
        with jsonlines.open(os.path.join(data_dir, "data.jsonl"), "r") as f:
            for sample in f:
                if sample["id"] in sample_indices:
                    evaluation_samples.append(sample)
                if len(evaluation_samples) == max_samples:
                    break
        return evaluation_samples

    def load_prompt_template(self, template_path):
        with open(template_path, "r") as f:
            return f.read()
    
    def load_fewshot_examples(self, data_dir, fewshot_filename, pool_size):
        """
            load fewshot examples
                default setup: few-shot step-by-step the demonstrations w.r.t each test program
                task var #1: few-shot are (program, input, trace) triplets not overlapped with test program
        """
        if self.fewshot_type == "input_trace":
            fewshot_data = {}
            with jsonlines.open(os.path.join(data_dir, fewshot_filename), "r") as f:
                for item in f:
                    item['inputs'] = item['inputs'][:pool_size]
                    item['traces'] = item['traces'][:pool_size]
                    fewshot_data[item["id"]] = item
            
            assert all(x['id'] in fewshot_data for x in self.evaluation_data)
            return fewshot_data

        elif self.fewshot_type == "program_input_trace":
            with open(os.path.join(data_dir, fewshot_filename), "rb") as f:
                fewshot_ids = pickle.load(f)[:pool_size]
            
            fewshot_examples = []
            with jsonlines.open(os.path.join(data_dir, "data.jsonl"), "r") as f:
                for item in f:
                    if item["id"] in fewshot_ids:
                        fewshot_examples.append(item)
            return fewshot_examples
        
    def eval(self):
        log_file = os.path.join(self.output_dir, "logs.jsonl")
        samples_to_evaluate = self.evaluation_data
        
        # skip already evaluated samples if log file exists
        if os.path.exists(log_file):
            evaluated_ids = set()
            with open(log_file, "r") as f:
                for line in f:
                    evaluated_ids.add(json.loads(line)["id"])
            samples_to_evaluate = [item for item in samples_to_evaluate if item["id"] not in evaluated_ids]
            
        # eval in parallel
        thread_map(
            self.eval_single_item, 
            samples_to_evaluate, 
            max_workers=self.num_workers, 
            chunksize=5
        )
        
    def eval_single_item(self, item):
        """evaluate a single test example"""
        prompts, responses, targets = self.get_responses(item)
        
        item["prompts"] = prompts
        item["responses"] = responses
        item["targets"] = targets

        del item["inputs"]
        del item["traces"]
        
        # handle dummy None client which returns empty responses
        if self.client is None or len(responses[0]) > 0:
            with open(os.path.join(self.output_dir, "logs.jsonl"), "a") as f:
                f.write(json.dumps(item)+"\n")
        
    def get_responses(self, item):
        """
            get model responses for a given test item.
                1. format program, input, and trace
                2. get few-shot examples string based on fewshot_type
                3. format prompt
                4. get model response
        """
        inputs, traces = item["inputs"], item["traces"]
        formatted_program = self.process_program(item["program"])
        formatted_inputs, formatted_traces = self.process_inputs_traces(inputs, traces, formatted_program)
        
        if self.fewshot_type == "input_trace":
            few_shot_str = self.format_fewshot_prompt_with_input_trace(item["id"])

        elif self.fewshot_type == "program_input_trace":
            few_shot_str = self.format_fewshot_prompt_with_program_input_trace()

        else:
            raise NotImplementedError("unsupported few-shot type")
        
        prompts = []
        responses = [] 
        for inp, trace in zip(formatted_inputs, formatted_traces):

            # format prompt
            this_prompt = self.format_prompt(
                prompt_template=self.prompt_template,
                test_program=formatted_program,
                test_input=inp,
                few_shot_str=few_shot_str,
            )

            # If client is None, we don't need to get actual responses
            if self.client is None:
                prompts.append(this_prompt)
                responses.append("")  # Empty response when client is None
                continue

            # get response
            max_toks_to_gen = len(self.tokenizer.encode("\n".join(trace[1:-1]))) + 200
            
            # adjust token limit for reasoning models
            if self.client.think:
                max_toks_to_gen = min(max_toks_to_gen * 20, self.client.max_length - len(self.tokenizer.encode(this_prompt)) * 2)
                if max_toks_to_gen < 0:
                    max_toks_to_gen = self.client.max_length 
                print(f"adjusted max requested tokens for reasoning models {max_toks_to_gen}")
            
            response = self.client(this_prompt, max_toks_to_gen)['text'][0]
            if response == "":
                print("warning: model returned empty string.")
            
            prompts.append(this_prompt)
            responses.append(response)
            
        return prompts, responses, formatted_traces
        
    def format_fewshot_prompt_with_program_input_trace(self):
        """
            format few-shot examples: (program, input, trace) triplets
        """
        prompt_template = (
            "Program:\n```\n{program}```\n\n"
            "Input:\n```\nfunction({input_args})\n```\n\n\n"
            "Output:\n```\n{trace_output}```\n"
        )
        
        selected_examples = random.sample(self.fewshot_examples, self.num_fewshot)
        few_shot_text = ""
        
        for example in selected_examples:
            formatted_program = self.process_program(example["program"])
            formatted_inputs, formatted_traces = self.process_inputs_traces(
                example["inputs"], 
                example["traces"], 
                formatted_program
            )
            
            example_text = prompt_template.format(
                program=formatted_program,
                input_args=formatted_inputs[0],
                trace_output=formatted_traces[0]
            )
            few_shot_text += example_text + f"\n{self.stop_token}\n\n"
            
        return few_shot_text
    
    def format_fewshot_prompt_with_input_trace(self, example_id):
        """
            format few-shot examples: (input, trace) pairs
        """
        prompt_template = (
            "\nInput:\n```\nfunction({input_args})\n```\n\n\n"
            "Output:\n```\n{trace_output}```\n"
        )
        
        # get test program and its examples
        program_examples = self.fewshot_examples[example_id]
        formatted_program = self.process_program(program_examples["program"])
        formatted_inputs, formatted_traces = self.process_inputs_traces(
            program_examples["inputs"], 
            program_examples["traces"], 
            formatted_program
        )
        
        # select random few-shot and format prompt
        few_shot_text = ""
        selected_indices = random.sample(range(len(formatted_inputs)), self.num_fewshot)
        
        for idx in selected_indices:
            example_text = prompt_template.format(
                input_args=formatted_inputs[idx],
                trace_output=formatted_traces[idx]
            )
            few_shot_text += example_text + f"\n{self.stop_token}\n"
            
        return few_shot_text
    
    def format_prompt(self, prompt_template, test_program, test_input, few_shot_str):
        """fill in the prompt template with program, input, and few-shot examples"""
        prompt = prompt_template.replace("{input_program}", test_program)
        prompt = prompt.replace("{fewshots}", few_shot_str)
        prompt = prompt.replace("{function_args}", test_input)
        return prompt
        
    def process_program(self, program):
        """add line numbers to program code"""

        program_lines = program.split("\n")
        line_numbers = [f"L{i}" for i in range(1, len(program_lines)+1)]
        formatted_program = "".join(
            f"{line_no} {code_line}\n" 
            for line_no, code_line in zip(line_numbers, program_lines)
        )
        return formatted_program
    
    def process_inputs_traces(self, inputs, traces, program):
        """format inputs and traces
        
        Args:
            inputs: List of input dictionaries for the program
            traces: List of execution trace lists
            program: program string
            
        Returns:
            tuple: (formatted_inputs, formatted_traces) where:
                - formatted_inputs: List of input strings in function call format
                - formatted_traces: List of execution trace strings
        """
        formatted_inputs = []
        formatted_traces = []
        
        for input_dict, trace_list in zip(inputs, traces):
            # format input as function call arguments
            input_args = ", ".join(
                f"{key}={value}" 
                for key, value in input_dict.items() 
                if key in program
            ).strip()
            

            # process trace excluding first and last lines
            # traces from generator contains initial and final states of all variables; use only intermediate states
            trace_lines = trace_list[1:-1]
            trace_output = ""
            
            for trace_line in trace_lines:
                # extract line info after first comma
                trace_info = trace_line[trace_line.index(",")+1:]
                line_no = "L" + trace_info[:trace_info.index(",")]
                
                # extract key-value pairs
                kv_pairs = trace_info[trace_info.index(",")+1:].split("#")
                
                # process key-value information
                if len(kv_pairs) > 1:
                    try:
                        # if tracer returned multiple kv pairs, extract the one with list-related values. 
                        # otherwise, just use the first one
                        kv_pairs = [x for x in kv_pairs if "lst" in x][0].split(":")
                    except:
                        kv_pairs = kv_pairs[0].split(":")
                else:
                    kv_pairs = kv_pairs[0].split(":")

                # format trace line:  line_no,variable_name:variable_value
                if len(kv_pairs) == 1:
                    trace_output += f"{line_no},\n"
                else:
                    key, value = kv_pairs[0], kv_pairs[1].replace(" ", "")
                    trace_output += f"{line_no},{key}:{value}\n"
            
            formatted_inputs.append(input_args)
            formatted_traces.append(trace_output)

        return formatted_inputs, formatted_traces


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./configs/XXX.yaml")
    parser.add_argument("--voter_id", type=int, default=0, help="voter id")
    args = parser.parse_args()
    
    
    with open(args.config_file, "r") as f:
        config = load_yaml_config(yaml.safe_load(f))
        config_version = os.path.splitext(os.path.basename(args.config_file))[0]

    # set fewshot_type
    prompt_fewshot_type_map = {
        "default.txt": "input_trace",                   # use `fewshots.jsonl`
        "task_variation_1.txt": "program_input_trace",  # use `fewshots_bin{x}.bin`
        "task_variation_2.txt": "input_trace",          # use `fewshots.jsonl`
        "zero_shot_v1.txt": "input_trace",
        "zero_shot_v2.txt": "input_trace",
        "zero_shot_v3.txt": "input_trace",
    }
    config.data.fewshot_type = prompt_fewshot_type_map[config.data.prompt_file]
    if config.data.fewshot_type == "program_input_trace":
        config.data.fewshot_file = f"fewshots_bin{config.data.data_split}.bin"

    config.data.voter_id = args.voter_id
    config.data.prompt_file = os.path.join("prompts", config.data.prompt_file)

    if config.model.model_name_or_path == "none":
        this_client = None
    else:
        this_client = Client(
            config=config.model,
        )
    
    try:
        this_tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    except:
        print(f"Unable to load tokenizer from path:\n {config.model.model_name_or_path}")
        this_tokenizer = tiktoken.get_encoding("o200k_base")

    # Initialize and run evaluator
    evaluator = Evaluator(
        client=this_client, 
        tokenizer=this_tokenizer,
        num_workers=config.data.num_workers,
        random_seed=config.data.voter_id,
        log_path=os.path.join(
            config.eval.result_folder,
            config_version,
            "results",
            f"voter_{config.data.voter_id}"
        ),
        data_path=config.data.folder,
        data_file=f"index_{config.data.split}.bin",
        data_size=config.data.size,
        num_fewshot=config.data.num_fewshot,
        fewshot_file=config.data.fewshot_file,
        fewshot_pool_size=config.data.fewshot_pool_size,
        fewshot_type=config.data.fewshot_type,
        prompt_file=config.data.prompt_file,
        stop=config.model.generation_kwargs.stop[0],
    )
    evaluator.eval()
