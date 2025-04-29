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
import ast
import sys
import copy
import uuid
import yaml
import json
import pickle
import random
import inspect
import argparse
import jsonlines
import numpy as np
from tqdm import tqdm
import multiprocessing
from dataclasses import dataclass
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

from grammar import *
from tracer import generate_ast_function_trace

# max number of attempts to generate valid inputs for a program 
MAX_AUGMENT_LOOPS = 10000

class ProgramState:
    """Tracks the state of program generation including variables and scope information.
    
    This class maintains information about:
        - Defined variables (integer, boolean, and list variables)
        - Maximum depth of program generation
        - Maximum indentation level
        - Stack of while loop conditions for context-sensitive generation
    """
    
    def __init__(self):
        """Initialize a new program state."""
        self.reset()
    
    def reset(self):
        self.defined_vars = []
        self.defined_lst_vars = []
        self.defined_bool_vars = []
        self.max_depth = 0
        self.max_indent = 0
        self.while_cond_stack = []
    
    def register_var(self, vars):
        self.defined_vars.extend(vars)
        self.defined_vars = list(dict.fromkeys(self.defined_vars))
    
    def register_bool_var(self, vars):
        self.defined_bool_vars.extend(vars)
        self.defined_bool_vars = list(dict.fromkeys(self.defined_bool_vars))
    
    def register_lst_var(self, list_vars):
        self.defined_lst_vars.extend(list_vars)
        self.defined_lst_vars = list(dict.fromkeys(self.defined_lst_vars))


class Generator:
    """generate simple Python programs based on specified grammar"""
    
    def __init__(self, args):
        self.args = args
        
        # can be overridden by args.overwritten_terminal
        self.terminal = {
            "<in_var>" : ["x", "y", "z", "u", "v", "w"],
            "<in_bool_var>": ["cond_x", "cond_y", "cond_z"],
            "<var>": ["i", "j", "k", "a", "b", "c", "m", "n"],
            "<bool_var>": ["cond_a", "cond_b", "cond_c", "cond_d", "cond_e"], 
            "<list_var>": ["lst_x", "lst_y", "lst_z", "lst_u", "lst_v", "lst_w"],
            "<number>" : [str(x) for x in range(args.max_num)],            
            "<while_cond_number>": [str(x) for x in range(4, args.max_while_loop, 2)],                 
            "<list_index_number>": [str(x) for x in range(int(args.max_list_len) // 2)],
            "<cnter_increment_number>": [str(x) for x in [2]],
            "<bool_value>": ["False", "True"],
            "<predefined>": ["if", "while", "True", ".append(", ".pop()", "len(", ")", "cnter",
                             ":", "=", "+", "-", ">", "<", "0", "\n",
                             "[", "]", "==", "!="]
        }
        self.program_state = ProgramState()
        self.production = self.load_grammar(args.grammar_ver)
        
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        
    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        
    def load_grammar(self, grammar_ver):
        try:
            data = GRAMMAR_MAP[grammar_ver]
        except:
            raise BaseException("Invalid grammar version")
            
        data = data.replace("\"\n\"", "nl").replace("\"", "")
        data = [x.strip().split("::=") for x in data[data.index("<program>"):].split("\n") if "::=" in x]
        grammar_dict = {x[0].strip(): [[y.strip().replace("nl", "\n") for y in z.split()] for z in x[1].split("|")] for x in data}
        grammar_dict["<min_stmt_lst>"] = [["<stmt>"] * int(self.args.min_code_lines * 0.4)]
        
        if hasattr(self.args, "overwritten_grammar"):
            try:
                overwritten = [x.strip().split("::=") for x in self.args.overwritten_grammar.split("@@")]
            except:
                raise BaseException("Invalid grammar error")
            for item in overwritten:
                grammar_dict[item[0].strip()] = [[y.replace("\"", "").replace("\\n", "\n") for y in x.strip().split()] for x in item[1].split("|")]
        
        if hasattr(self.args, "overwritten_terminal"):
            try:
                overwritten = [x.strip().split("::=") for x in self.args.overwritten_terminal.split("@@")]
            except:
                raise BaseException("Invalid grammar error")
            
            for item in overwritten:
                self.terminal[item[0].strip()] = [x.strip().split()[0] for x in item[1].split("|")]
                
        return grammar_dict

    def generate_program(self, num_integer_var=1, num_lst_var=1, num_bool_var=1):
        # define input_variables
        init_var_maps = self._initialize(num_integer_var, num_lst_var, num_bool_var) 
        while True:
            program = self._generate("<program>", None)
            if program:
                break
        program = self._post_process(init_var_maps[0], program)
        return init_var_maps, program, self.program_state

    def _initialize(self, num_integer_var=1, num_lst_var=1, num_bool_var=1):
        assert num_integer_var <= len(self.terminal["<in_var>"])
        assert num_bool_var <= len(self.terminal["<in_bool_var>"])
        assert num_lst_var <= len(self.terminal["<list_var>"])

        input_variables = random.sample(self.terminal["<in_var>"], k=num_integer_var)
        input_bool_variables = random.sample(self.terminal["<in_bool_var>"], k=num_bool_var)
        input_lst_variables = random.sample(self.terminal["<list_var>"], k=num_lst_var)
        
        # register init variables to enforce stateful update
        # program states now only track minimally the correctedness of program
        self.program_state.register_var(input_variables)
        self.program_state.register_lst_var(input_lst_variables)
        self.program_state.register_bool_var(input_bool_variables)

        init_var_maps = []
        # sample multiple inputs for the same program
        for _ in range(self.args.num_inputs_per_program):
            init_vars = {}
            for input_var in input_variables:
                init_vars[input_var] = int(random.choice(self.terminal['<number>']))
            for input_var in input_lst_variables:
                init_vars[input_var] = [int(random.choice(self.terminal['<number>'])) for _ in range(random.randint(self.args.min_list_len, self.args.max_list_len))]
            for input_var in input_bool_variables:
                init_vars[input_var] = random.choice([True, False])
            init_var_maps.append(init_vars)
        
        return init_var_maps
    
    def _generate(self, symbol, parent, pos=0, depth=0, indent=0):
        """
            recursively expand production rules
        """

        if depth > self.args.max_expansion_depth:
            return None

        self.program_state.max_depth = max(depth, self.program_state.max_depth)
        self.program_state.max_indent = max(indent, self.program_state.max_indent)

        # terminals: special cases
        if symbol == "<number>":
            if parent == "<increment_cnter>":
                return random.choice(self.terminal["<cnter_increment_number>"])
        
        if symbol == "<var>":
            if parent == "<num_assignment>" and pos == 0:
                return random.choice(self.terminal["<in_var>"] + self.terminal["<var>"])
            return random.choice(self.program_state.defined_vars)
        
        if symbol == "<bool_var>":
            if parent == "<bool_assignment>" and pos == 0:
                return random.choice(self.terminal["<in_bool_var>"] + self.terminal["<bool_var>"])
            return random.choice(self.program_state.defined_bool_vars)
        
        if symbol == "<list_var>":
            return random.choice(self.program_state.defined_lst_vars)
        
        # while_block csg
        if symbol.startswith("<while_cond_var"):
            if ",1>" in symbol:
                if self.program_state.while_cond_stack:
                    used_cond_var = [x for lst in self.program_state.while_cond_stack for x in lst if "=" not in x]
                else:
                    used_cond_var = []
                candidate_cond_vars = set(self.terminal["<in_bool_var>"] + self.terminal["<bool_var>"]) - set(used_cond_var)
                if len(candidate_cond_vars) == 0:
                    return None
                cond_var = random.choice(sorted(list(candidate_cond_vars)))
                # store while_cond_var to stack for later reuse
                self.program_state.while_cond_stack.append([cond_var])
                return cond_var

            elif ",2>" in symbol:
                if self.program_state.while_cond_stack:
                    return self.program_state.while_cond_stack[-1][0]
                else:
                    return None

            else:
                return None

        # terminals: common cases
        if symbol in self.terminal:
            return random.choice(self.terminal[symbol])
        
        if symbol in self.terminal["<predefined>"]:
            if "cnter" in symbol:
                if parent == "<increment_cnter>":
                    symbol = symbol.replace("cnter", f"cnter_{indent-1}")
                else:
                    symbol = symbol.replace("cnter", f"cnter_{indent}")
            return symbol

        # context-sensitive non-terminals
        if symbol == "<while_cond,2>":
            if len(self.program_state.while_cond_stack) == 0:
                return None
            this_while_cond = self.program_state.while_cond_stack[-1][-1]
            cnter_indent = int(this_while_cond[this_while_cond.index("cnter_")+6:this_while_cond.index("!=")].strip())
            if cnter_indent+1 != indent:
                return None
            return "\t"*indent + self.program_state.while_cond_stack[-1][-1]
        
        # non-terminals
        try:
            assert symbol in self.production
        except:
            raise BaseException(f"Invalid symbol {symbol} not in production rules.")
        
        productions = self.production[symbol]
        random.shuffle(productions)

        if ["<while_block_nh>"] in productions:
            productions.remove(["<while_block_nh>"])
            if np.random.random() < 0.05:
                productions.append(["<while_block_nh>"])

        for production in productions:
            parts = []

            # expansion
            for si, this_sym in enumerate(production):
                next_indent = indent
                if symbol in ["<if_block>", "<while_block>", "<while_block_nh>"]:
                    if this_sym in ["<stmt_lst>", "<increment_cnter>", "<while_cond,2>"]:
                        next_indent = indent + 1

                if next_indent > self.args.max_scope_depth:
                    parts.append(None)
                    break   

                if symbol in ["<num_assignment>", "<bool_assignment>",
                              "<increment_cnter>", "<list_op>", "<if_block>"] and si == 0:
                    parts.append("\t"*indent)
                
                if symbol == "<while_block>" and si in [0, 1, 2] :
                    parts.append("\t"*indent)
                
                if symbol == "<while_block_nh>" and si in [0]:
                    parts.append("\t"*indent)
                
                part = self._generate(this_sym, 
                                    parent=symbol, 
                                    pos=si, 
                                    depth=depth+1, 
                                    indent=next_indent)

                parts.append(part)

            if (None in parts) or all(len(x.strip()) == 0 for x in parts):
                # print(depth, "None in parts", parts)
                continue
            else:
                if symbol == "<while_cond,1>":
                    cond_expr = None if None in parts else " ".join(parts)
                    # push condition expression for later reuse
                    self.program_state.while_cond_stack[-1].append(cond_expr)

                if symbol == "<while_block>":
                    # pop condition once a while block closes
                    self.program_state.while_cond_stack.pop()


            parts = [x for x in parts if len(x) > 0]
            if symbol == "<num_assignment>":
                first_var = [x for x in parts if len(x.strip()) > 0][0]
                self.program_state.register_var([first_var])
            
            if symbol == "<bool_assignment>":
                first_var = [x for x in parts if len(x.strip()) > 0][0]
                self.program_state.register_bool_var([first_var])
            
            ret = " ".join([x for x in parts if len(x) > 0])

            if symbol == "<stmt>":
                ret += "\n"
            return ret
        
        return None


    def _post_process(self, init_vars, program):
        if program:
            # add function definition
            input_args = ", ".join(list(init_vars.keys()))
            func_def = "def function(" + input_args.strip() + "):\n"
            program = program.replace("\n", "\n\t")
            program = program.replace(" .pop", ".pop") \
                         .replace(" .append", ".append") \
                         .replace(" :", ":") \
                         .replace("\n ", "\n") \
                         .replace(" [ ", "[") \
                         .replace(" ]", "]") \
                         .replace("( ", "(") \
                         .replace(" )", ")") \
                         .replace("\n ", "\n") \
                         .replace("\t ", "\t") \
                         .replace("  cnter", "cnter") 
            program = [x for x in program.split("\n") if len(x.strip()) != 0]
            program = "\n".join(program)
        return func_def + "\t" + program + "\n\t" + "return"


@dataclass
class ERROR_CODES:
    """for tracking errors in program generation & input validation"""
    SUCCESS: int = 0
    DUPLICATES: int = 1
    SHORT_PROGRAM: int = 2
    EXEC_ERROR: int = 3
    LONG_PROGRAM: int = 4
    NONE_INPUTS: int = 6

class ProgramGenerationError(Exception):
    """Base exception for program generation errors."""
    pass

class GrammarError(ProgramGenerationError):
    """Exception raised for grammar-related errors."""
    pass

class ValidationError(ProgramGenerationError):
    """Exception raised for program validation errors."""
    pass

def check_program(args, input_lst, program, signature, sigs, action="generate"):
    """validate generated program & traces"""
    if not program or not input_lst:
        return ERROR_CODES.NONE_INPUTS, None
        
    # check redundancy
    if signature in sigs:
        return ERROR_CODES.DUPLICATES, None

    # check number of lines
    program_lines = [x for x in program.split("\n") if len(x.strip()) > 0]
    if len(program_lines) <= args.min_code_lines:
        return ERROR_CODES.SHORT_PROGRAM, None

    if len(program_lines) >= args.max_code_lines:
        return ERROR_CODES.LONG_PROGRAM, None
    
    # check executable, get trace
    func_ast = ast.parse(program)
    _, traces = generate_ast_function_trace(func_ast, input_lst, max_steps=args.max_trace_steps)

    expected_num_traces = args.num_inputs_per_program if action == "generate" else 1 # step 1 allows for multiple inputs per program; step 3 augments single input each time
    if len(traces) != expected_num_traces:
        return ERROR_CODES.EXEC_ERROR, None
        
    return ERROR_CODES.SUCCESS, traces

def extract_function_args(program):
    """extract function arguments from program definition"""
    input_program = program.split("\n")
    func_def = input_program[0]
    if not func_def.startswith("def function("):
        raise ValidationError("Invalid program format: missing function definition")
        
    args_str = func_def.replace("def function(", "").replace("):", "")
    return [x.strip() for x in args_str.split(",")]

def is_arg_used(arg, program_code):
    """check if an argument is used in the program code"""
    if "lst" in arg:
        return f"{arg}" in program_code
    return (f" {arg}" in program_code or 
            f"({arg}" in program_code or 
            f"[{arg}" in program_code)

def rm_irrelevant_arg(input_lst, program):
    """remove unused arguments from program & input list"""
    try:
        # extract and filter arguments
        func_args = extract_function_args(program)
        program_code = "\n".join(program.split("\n")[1:])
        
        # keep only used arguments
        used_args = [arg for arg in func_args if is_arg_used(arg, program_code)]
        
        # update function definition
        input_program = program.split("\n")
        input_program[0] = f"def function(" + ", ".join(used_args) + "):"
        
        # filter input list
        new_input_lst = []
        for inputs in input_lst:
            filtered_inputs = {k: inputs[k] for k in inputs if k in used_args}
            new_input_lst.append(filtered_inputs)
            
        return new_input_lst, "\n".join(input_program)
        
    except Exception as e:
        raise ValidationError(f"Failed to remove irrelevant arguments: {str(e)}")

def generate_batch_data(args, generator, sigs, error_cnter, start_idx, end_idx, ret_lock, process_id):
    process_seed = args.random_seed + process_id
    generator = Generator(args)
    generator.set_random_seed(process_seed)
    
    ret = []
    while len(ret) <  (end_idx - start_idx):
        generator.program_state.reset()

        num_input_var = random.randint(1, len(generator.terminal["<in_var>"]))
        num_input_lst = random.randint(2, len(generator.terminal["<list_var>"]))
        num_input_bool = random.randint(1, len(generator.terminal["<in_bool_var>"]))
        input_lst, program, prog_state = generator.generate_program(
                                            num_integer_var=num_input_var,
                                            num_lst_var=num_input_lst,
                                            num_bool_var=num_input_bool,
                                        )
        input_lst, program = rm_irrelevant_arg(input_lst, program)

        if len(input_lst) == 0 or len(input_lst[0]) == 0:
            error_code = 6
            error_cnter[error_code] += 1
            continue

        signature = str(uuid.uuid3(uuid.NAMESPACE_DNS, program))
        _input_lst = copy.deepcopy(input_lst)   
        error_code, traces = check_program(args, _input_lst, program, signature, sigs)
        error_cnter[error_code] += 1
        if traces:
            traces = [trace.split("\n") for trace in traces]
            item = {
                "id": signature,
                "program": program,
                "num_inputs": num_input_var + num_input_lst + num_input_bool,
                "num_lines": len([x for x in program.split("\n") if len(x.strip()) > 0]),
                "max_expand_depth": prog_state.max_depth,
                "max_scope_depth": prog_state.max_indent,
                "inputs": input_lst,
                "traces": traces,
                "num_traces_line": [len(x) for x in traces],
                "num_traces_chars": [sum([len(trace_line) for trace_line in trace]) for trace in traces]
            }
            ret.append(item)
            sigs[signature] = True

    with ret_lock:
        return ret, error_cnter, sigs


def generate_data(args):
    error_cnter = defaultdict(int)
    generator = Generator(args)
    ret = []

    with multiprocessing.Manager() as manager:
        sigs = manager.dict()
        ret_lock = manager.Lock()
        
        num_processes = getattr(args, 'num_processes', multiprocessing.cpu_count())
        results = []

        total_iterations = args.generate_limit
        chunk_size = total_iterations // num_processes
        
        with multiprocessing.Pool(num_processes) as pool:
            results = []

            for i in range(num_processes):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < num_processes - 1 else total_iterations
                
                results.append(pool.apply_async(generate_batch_data, 
                                               (args, generator, sigs, error_cnter, start_idx, end_idx, ret_lock, i)))
            
            with tqdm(total=total_iterations) as pbar:
                # collect from all processes
                for result in results:
                    data, error_cnter, sigs = result.get()
                    with ret_lock:  
                        ret.extend(data)
                    pbar.update(len(data)) 


    # write the result to files
    fpath = os.path.join(args.output_path, args.grammar_ver, args.config_ver)
    os.system(f"mkdir -p {fpath}")

    with jsonlines.open(os.path.join(fpath, "data.jsonl"), "w") as f:
        f.write_all(ret)

    with open(os.path.join(fpath, "generate.stats"), "w") as f:
        error_dict = {}
        for error in ERROR_CODES.__dataclass_fields__:
            error_dict[ERROR_CODES.__dataclass_fields__[error].default] = error
        f.write("\n")
        num_total = sum(error_cnter.values())
        f.write("Error ratio:\n")
        for k in error_cnter:
            f.write(f"{error_dict[k]}: {np.round(error_cnter[k] / num_total, 2)}\n")
        traces_len = np.array([x["num_traces_line"] for x in ret])
        f.write(f"Average traces lines: {traces_len.mean()}\n")
        f.write(f"Max traces lines: {traces_len.max()}\n")
        f.write("Trace lines histogram:\n")
        val, hist = np.histogram(traces_len)
        f.write("count" + np.array2string(val) + "\n")
        f.write("threshold" + np.array2string(hist) + "\n")
        # write a few sampled program
        for idx in random.choices(range(0, len(ret)), k=5):
            f.write("="*23 + "\n")
            f.write(ret[idx]['program'] + "\n\n")

def sample_data(args):
    # assert file exists    
    fpath = os.path.join(args.output_path, 
                         args.grammar_ver, 
                         args.config_ver)
    assert os.path.exists(os.path.join(fpath, "data.jsonl"))

    # produce indices
    res = []
    with jsonlines.open(os.path.join(fpath, "data.jsonl"), "r") as f:
        for i, item in enumerate(f): 
            res.append((np.mean(item['num_traces_line']), item))
    
    traces_len = [x[0] for x in res]
    traces_max, traces_min = np.max(traces_len), np.min(traces_len)
    if not hasattr(args, "num_bins"):
        args.num_bins = 10

    thresholds = np.arange(traces_min*1.5, traces_max*0.8, (traces_max*0.8-traces_min*1.5)/(args.num_bins - 1))
    bin_data = defaultdict(list)
    fewshot_bin_data = defaultdict(list)

    # separate test and few-shot examples (few-shot for task variation #1 in paper)
    # in default setup, few-shot are step-by-step demonstrations in data_augment.jsonl
    for item in res:
        bin_index = np.digitize(item[0], thresholds, right=True) 
        bin_index += 1
        if len(bin_data[bin_index]) < args.sample_bin_limit:
            bin_data[bin_index].append(item[1]['id'])
        else:
            fewshot_bin_data[bin_index].append(item[1]['id'])
    
    with open(os.path.join(fpath, "sample.stats"), "w") as f:
        f.write("bin_index: count\n")
        f.write(str({k: len(bin_data[k]) for k in bin_data}) + "\n")

    for k in bin_data:
        with open(os.path.join(fpath, f"index_{k}.bin"), "wb") as f:
            pickle.dump(bin_data[k], f)

    # fewshots indices grouped by trace size
    for k in fewshot_bin_data:
        with open(os.path.join(fpath, f"fewshots_bin{k}.bin"), "wb") as f:
            pickle.dump(fewshot_bin_data[k], f)

    fewshots = []
    all_test_ids = sorted(list(set([item for k in bin_data for item in bin_data[k]])))
    for item in res:
        item = item[1]
        if item['id'] not in all_test_ids:
            fewshots.append(item['id'])

    # fewshots indices not grouped
    with open(os.path.join(fpath, f"fewshots.bin"), "wb") as f:
        pickle.dump(fewshots, f)

def gen_new_input_map(generator, input_eg, args):
    """Generate a new input map based on example input."""
    input_map = {}
    for k in input_eg:
        if "lst" in k:
            input_map[k] = [int(random.choice(generator.terminal['<number>'])) 
                          for _ in range(random.randint(args.min_list_len, args.max_list_len))]
        elif "cond" in k:
            input_map[k] = random.choice([True, False])
        else:
            input_map[k] = int(random.choice(generator.terminal['<number>']))
    return input_map

def is_duplicate_input(new_input, existing_inputs):
    """check if an input map is a duplicate of existing ones"""
    return any(new_input == existing for existing in existing_inputs)

def augment_single_data(input):
    """generate augmented data for a single program example"""
    i, item, generator, args, fpath, process_id = input
    process_seed = args.random_seed + process_id
    generator = Generator(args)
    generator.set_random_seed(process_seed)
    
    program = item["program"]
    input_eg = item["inputs"][0]
    max_loop = MAX_AUGMENT_LOOPS

    lst_input_maps = []
    lst_traces = []
    
    while len(lst_input_maps) != args.num_aug_inputs_per_program and max_loop > 0:
        new_input_map = gen_new_input_map(generator, input_eg, args)
        
        # skip if duplicate
        if is_duplicate_input(new_input_map, lst_input_maps):
            max_loop -= 1
            continue
            
        if is_duplicate_input(new_input_map, item["inputs"]):
            max_loop -= 1
            continue
            
        # generate trace, need to deepcopy input_map, which will be modified by check_program
        new_input_map_copy = copy.deepcopy(new_input_map)
        error_code, traces = check_program(args, [new_input_map_copy], program, item['id'], {}, action="augment")
        
        if traces is not None:
            lst_input_maps.append(new_input_map)
            lst_traces.extend(traces)
            
        max_loop -= 1

    if max_loop == 0:
        print(f"skip example {i}. Hit max tries limit.")
        return None
        
    # create augmented item
    traces = [trace.split("\n") for trace in lst_traces]
    augmented_item = {
        "id": item["id"],
        "program": program,
        "num_inputs": item["num_inputs"],
        "num_lines": len([x for x in program.split("\n") if len(x.strip()) > 0]),
        "max_expand_depth": item["max_expand_depth"],
        "max_scope_depth": item["max_scope_depth"],
        "inputs": lst_input_maps,
        "traces": traces,
        "num_traces_line": [len(x) for x in traces],
        "num_traces_chars": [sum([len(trace_line) for trace_line in trace]) for trace in traces]
    }
    
    return i, augmented_item

def augment_data(args):
    # assert file exists    
    fpath = os.path.join(args.output_path, 
                         args.grammar_ver, 
                         args.config_ver)
    assert os.path.exists(os.path.join(fpath, "data.jsonl"))
    augment_file = os.path.join(fpath, "fewshots.jsonl")
    if os.path.exists(augment_file):
        os.remove(augment_file)

    generator = Generator(args)
    all_data = []
    with jsonlines.open(os.path.join(fpath, "data.jsonl"), "r") as f:
        for item in f:
            all_data.append(item)

    if hasattr(args, "augment_test_only") and args.augment_test_only:
        # load sampled index and only augment those
        bin_indices_fname = [fn for fn in os.listdir(fpath) if fn.startswith("index_") and fn.endswith(".bin")]
        all_ids = []
        for bin_fn in bin_indices_fname:
            with open(os.path.join(fpath, bin_fn), "rb") as f:
                data = pickle.load(f)
            all_ids.extend(data)
        
        all_ids = sorted(list(set(all_ids)))
        all_data = [x for x in all_data if x["id"] in all_ids]

    print(f"Augmenting {len(all_data)} problems each with {args.num_aug_inputs_per_program} (input, trace) pairs..")

    # multiprocessing
    num_processes = getattr(args, 'num_processes', multiprocessing.cpu_count())
    process_data = [(i, item, generator, args, fpath, i % num_processes) 
                   for i, item in enumerate(all_data)]
    results = process_map(augment_single_data, process_data, max_workers=num_processes)
    
    valid_results = [r for r in results if r is not None]
    valid_results.sort(key=lambda x: x[0])  
    
    # write results in order
    with open(os.path.join(fpath, "fewshots.jsonl"), "w") as f:
        for _, item in valid_results:
            f.write(json.dumps(item) + '\n')


class Config:
    def __init__(self, data):
        self.__dict__.update(data)

if __name__ == "__main__":

    yaml_path = sys.argv[1]
    with open(yaml_path, "r") as f:
        config = Config(yaml.safe_load(f))
    
    config_path = os.path.basename(yaml_path)
    config.config_ver = os.path.splitext(config_path)[0]
    
    ACTIONS = {
        'generate': generate_data,      # step 1. generate test examples, i.e., (program, input, trace) triplets, based on specified grammar and input yaml file
        'sample': sample_data,          # step 2. sample and group test examples based on target trace length into different bins; separate test triplets from the potential few-shot examples for task variation #1 in the paper
        'augment': augment_data,        # step 3. augment test examples with new (input, trace) pairs, for programs in data.jsonl, or test programs only if args.augment_test_only
    }
    for action in config.actions:
        ACTIONS[action](config)