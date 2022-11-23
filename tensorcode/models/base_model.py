# must use Python 3.10+
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tensorcode._utils.export_helpers import export, ContextManaged
from tensorcode._utils.fp import Test
from tensorcode._utils.overloaded import Overloaded
import tensorcode as tc
from tensorcode.utils.annotations import like, enc


@ContextManaged
class Model(tc.utils.annotations.SupportsEncode[any]):
    """I don't want to put so many methods in the same file, but I don't see a better
    way to group all the call signatures togethor. Also, many of these methods call each other.
    Finally, it just makes sense for LLM-based methods to all enter context simultaneously.

    I should group the items in this class into sub-class/modules so it looks like a real module
    even though its a class
    """
    
    @export
    @Overloaded
    def encode(self,
        object: any,
        context: any,
        instructions: enc[str],
        focus: like[str],
        examples: list):
        pass
    # in child classes, directly write
    # @encode.overload(x -> x is Number)
    # def encode_number(self, number): ...
    # this is possible since python classes are basically a module

    @export
    @Overloaded
    def create(description: R[str] | like[str], type: type|R[type], max_depth=1, recursions=1):
        pass

    @export
    @Overloaded
    # create bool with prompt engineered for decision:
    def decide(question: str):
        pass

    @export
    @Overloaded
    def write(prompt: str):  # create str with prompt engineered for writing:
        pass

    @export
    @Overloaded
    def define(description: str):  # define class:
        pass

    @export
    @Overloaded
    def get(obj: any, n: int | str, description: str, max_depth=0) -> any:
        pass

    @export
    @Overloaded
    def put(obj: any, val: any, description: str, max_depth=0):
        pass

    @export
    @Overloaded
    def sort(items: list, criteria: str) -> list:
        pass

    @export
    @Overloaded
    def similarity(a: any, b: any) -> float:
        pass

    @export
    @Overloaded
    def closest(item: T, choices: set[T]) -> T:  # find the closest match:
        pass

    @export
    @Overloaded
    def identify_anomalies(examples: set) -> set:
        pass

    @export
    @Overloaded
    def identify_patterns(examples: set, n=None, min=None, max=None, recognition_threshold=None) -> Pattern:
        pass

    @export
    @Overloaded
    def identify_isomorphisms(examples: set) -> set[tuple[Pattern, set]]:
        pass

    @export
    @Overloaded
    # organize items into groups (number of groups is auto-detected if not specified):
    def group(items, n=None, min=None, max=None):
        pass

    @export
    @Overloaded
    def predict(seq, steps=1):  # predict next n elements in sequence:
        pass

    @export
    @Overloaded
    # produces an item with the average appearance, shape, value, etc of the inputs:
    def average(items):
        pass

    @export
    @Overloaded
    # produces an item with a semantic value equaling the combination of the inputs:
    def combine(items):
        pass

    @export
    @Overloaded
    def _if(condition1: bool, fn1, condition2: bool, fn2, ..., conditionN: bool, fnN, else_fn):
        pass

    @export
    @Overloaded
    def _elif(condition: bool, fn):
        pass

    @export
    @Overloaded
    def _else(fn):
        pass

    @export
    @Overloaded
    # coconut is useful here:
    def switch(condition: bool, cases: dict, default=None)
    pass

    @export
    @Overloaded
    def _while(condition: bool, loop):
        pass

    @export
    @Overloaded
    def call(fn)  # produced wrapped function that intelligently selects parameters when called. You should pre-curry mandatory arguments and initiate the call afterwards if you don't have any conditioning parameters to supply. Useful as a wrapper for auto-converting inputs to the proper type. EG, tc._if asks for a bool, but you can give it a string and it will convert it to a bool automatically. Also, call avoids decoding from R only to have the function re-encode it to R in the backend.:
    pass

    @export
    @Overloaded
    # executes a series of operations (including tensorcode operations) and returns result. The central function for tensorcode-based interpreter:
    def do(instructions: R) -> any?:
        pass

    @export
    @Overloaded
    # returns {'clusters': ..., 'trends': ..., 'statistics': ..., notes: ['blah blah blah', ...], ...}:
    def analyze(items):
        pass

    @export
    @Overloaded
    # auto-generates visualizations that may be useful to understand the object(s) (code, data, or class definitions) and runs them. Useful in an interactive computing environment:
    def visualize(object):
        pass

    @export
    @Overloaded
    @tensorcode.call  # TODO: how can I use call to route for the call function?????
    def write_tests(self,
                    function: Callable,
                    instructions: str,
                    n: int = 1,
                    intended_behavior: str = None,
                    examples: list[Test] = None) -> list[Test]:
        '''auto-generates tests that may be useful to run and runs them. 
        Useful in an interactive computing environment'''
        pass

    @export
    @Overloaded
    def benchmark(function: Callable, instructions: str, metrics: list[str] = None):
        pass

    @export
    @Overloaded
    def debug(self, program: str | Callable):
        """
        program:
          - path to program
          - 
        """
        # uses 3rd party tools to decompose program into modules of interest, tests, their expected behavior, and their actual behavior. Based on test result divergence, makes appropriate changes to code. This should be a future todo rather than main item:
        pass

    # parameters are auto-encoded to R's or decoded to specific types as needed using the @tensorcode.call decorator
    # TODO: most functions should have these extra params
    # focus refers to which parts of the objects should be focused on
    # context is just extra information
    # instructions
    # examples

    @export
    def instruct(self, instructions: list[str]): ...
    @export
    def add_loss(self, loss, info, operation): ...
    @export
    def add_metric(self, metric, info, operation): ...
