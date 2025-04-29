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

"""
Implements a tool to trace the steps of a function.

Usage:
    from tracer import generate_ast_function_trace

    generate_ast_function_trace(ast_of_function, input_values, max_steps)
    
"""
import ast
import sys
import types
import copy
import dis
from collections import defaultdict
from typing import NamedTuple

class TraceLine(NamedTuple):
    line_number: int
    step_counter: int
    variable_state: dict
    event: str = ""

class MaxStepsException(Exception):
    pass

class LineCategories(NamedTuple):
    loaded_variables: dict
    stored_variables: dict
    is_pop_or_append: dict

class ProgramTracer:
    """trace a single call to a function with name function name"""
    def __init__(self, function_name, max_steps):
        self.function_name = function_name
        self.max_steps = max_steps
        self.run_trace = []
        self.last_line_executed = None
        self.step_counter = 0
        self._in_function = False

    def named_function_tracer(self):
        def _trace_named_func(frame, event, arg):
            nonlocal self

            if event == 'exception':
                return None

            if event == 'call':
                assert frame.f_code.co_name == self.function_name

                self.step_counter = 0
                self._in_function = True

                local_vars = copy.deepcopy(frame.f_locals)
                self.new_trace_line(
                    self.step_counter, frame.f_lineno, local_vars, event=event
                )
                self.step_counter += 1
                return _trace_named_func

            elif event in ('return', 'line') and self._in_function:
                local_vars = copy.deepcopy(frame.f_locals)
                self.new_trace_line(
                    self.step_counter, frame.f_lineno, local_vars, event=event
                )
                self.step_counter += 1
                if self.step_counter > self.max_steps:
                    raise MaxStepsException(self.step_counter)
                return _trace_named_func

        return _trace_named_func

    def new_trace_line(self, step_counter, line_number, local_vars, event):
        self.run_trace.append(
            TraceLine(
                step_counter=step_counter,
                line_number=line_number,
                variable_state=local_vars,
                event=event
            )
        )

def generate_trace(func, input_values, max_steps=200):

    pt = ProgramTracer(func.__name__, max_steps=max_steps)
    old_trace = sys.gettrace()
    sys.settrace(pt.named_function_tracer())

    try:
        func(**input_values)
    except MaxStepsException as e:
        pass
    except Exception as e:
        return None
    finally:
        sys.settrace(old_trace)

    return pt.run_trace

def disassemble_bytecode(func):
    # map bytecode operations to line numbers with disassembler
    is_pop_or_append = defaultdict(lambda: False)
    loaded_variables = defaultdict(list)
    stored_variables = defaultdict(list)
    
    starts_line = 0
    for b in list(dis.Bytecode(func)):
        if b.starts_line is not None:
            starts_line = b.starts_line

        if 'LOAD_FAST' in b.opname:
            loaded_variables[starts_line].append(b.argval)
        if 'STORE_FAST' in b.opname:
            stored_variables[starts_line].append(b.argval)
        try:
            if 'LOAD_ATTR' in b.opname and b.argval in ('pop', 'append'):
                is_pop_or_append[starts_line] = True
        except:
            try:
                if 'LOAD_METHOD' in b.opname and b.argval in ('pop', 'append'):  # older python versions
                    is_pop_or_append[starts_line] = True
            except:
                raise Exception(f"Error analyzing bytecode for line {starts_line}: {b}")

    return LineCategories(loaded_variables=loaded_variables, stored_variables=stored_variables, is_pop_or_append=is_pop_or_append)

def filter_trace(trace_lines, func):

    trace_line_categories = disassemble_bytecode(func)
    ret = [trace_lines[0]]
    
    for t in trace_lines[1:]:
        line_number = t.line_number
        
        print_loaded_lists = trace_line_categories.is_pop_or_append[line_number]
        print_all = t.event == 'return' # print all variables at the last line; this can be used to evaluate "outcome" of a function though we focus on procedural crrctness in L0-bench
        
        ret.append(
            TraceLine(
                line_number=t.line_number, 
                step_counter=t.step_counter,
                variable_state={k: v for k, v in t.variable_state.items() if
                                   k in trace_line_categories.stored_variables[t.line_number]
                                   or (print_loaded_lists and k in trace_line_categories.loaded_variables[t.line_number]) or print_all},
                event=t.event
            ))

    return ret

def post_process_trace(trace_lines):
    ret = [trace_lines[0]]
    for curr_l, next_l in zip(trace_lines[1:], trace_lines[2:]):
        ret.append(
            TraceLine(
                line_number=curr_l.line_number,
                step_counter=curr_l.step_counter,
                variable_state=next_l.variable_state,
                event=curr_l.event
            ))

    if trace_lines[-1].event == 'return':
        ret.append(trace_lines[-1])
    return ret

def format_trace(trace_lines, input_value, func_object, func_representation, func_ast):
    """format trace lines: <step>,<line index>,<updated_variables>"""
    aligned = post_process_trace(trace_lines)               # align variable state and executed line number
    filtered_result = filter_trace(aligned, func_object)   # filter variables based on whether they're stored, loaded in list operations

    string_trace = []
    for tl in filtered_result:
        # only store list when multiple variables exist
        if len(tl.variable_state.items()) <= 1:
            var_rep = ",".join([f"{k}:{v}" for k, v in tl.variable_state.items()])
        else:
            var_rep = ",".join([f"{k}:{v}" for k, v in tl.variable_state.items() if "lst" in k])
        string_trace.append(f"{tl.step_counter},{tl.line_number},{var_rep}")
    return string_trace

def generate_ast_function_trace(ast_of_function, input_values, max_steps):
    """
    traces the execution of a given function with specified input values.
    
    args:
        ast_of_function: the AST of the function to trace
        input_values: list of input value dictionaries
        max_steps: max number of execution steps to trace, exception is raised if exceeded
    """

    representation_of_function = ast.unparse(ast_of_function)
    ast_of_function = ast.parse(representation_of_function)
    
    local_variables_for_eval = {}
    eval(compile(ast_of_function, filename="<string>", mode='exec'), globals(), local_variables_for_eval)
    func_object = list(local_variables_for_eval.values())[0]
    
    formatted_traces = []
    for input_value in input_values:

        # generate raw trace
        trace_lines = generate_trace(func_object, input_value, max_steps)
        if trace_lines is None:
            continue
        
        # format trace
        formatted_trace = format_trace(trace_lines, input_value, func_object, representation_of_function, ast_of_function)
        formatted_traces.append(formatted_trace)

    string_traces = ["\n".join(trace_lines) for trace_lines in formatted_traces]
    return representation_of_function, string_traces
