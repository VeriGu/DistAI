# this file provides some low-level functions called by translate.py
# our design choice is that this file should not access any global variables in translate.py

import re
from ivy_parser import *


def separate_ivy_bexpr_by_and(ivy_bexpr):
    # this is incorrect when -> exists
    parenthesis_count = 0
    separators = []
    for i, c in enumerate(ivy_bexpr):
        if c == '(':
            parenthesis_count += 1
        elif c == ')':
            parenthesis_count -= 1
        elif parenthesis_count == 0 and c == '&':
            separators.append(i)
    sub_ivy_bexprs = []
    last = 0
    for separator in separators:
        sub_ivy_bexprs.append(ivy_bexpr[last:separator].strip())
        last = separator + 1
    sub_ivy_bexprs.append(ivy_bexpr[last:].strip())
    return sub_ivy_bexprs


def translate_remove_le(ivy_expr):
    curr_expr = ivy_expr
    match = re.search('([^a-zA-Z0-9_]|^)le\(', curr_expr)
    while match is not None:
        right_parenthesis_idx = find_closing_parenthesis(curr_expr, match.end() - 1)
        comma_idx = -1
        count = 0
        for i in range(match.end(), len(curr_expr)):
            if curr_expr[i] == ',' and count == 0:
                comma_idx = i
                break
            if curr_expr[i] == '(':
                count += 1
            elif curr_expr[i] == ')':
                count -= 1
        assert(match.end() < comma_idx < len(curr_expr) - 1)
        match_start = match.start() if match.group(0) == 'le(' else match.start() + 1  # handle start-of-string ^
        curr_expr = '{}({} <= {}){}'.format(curr_expr[:match_start], curr_expr[match.end(): comma_idx],
                                            curr_expr[comma_idx + 1: right_parenthesis_idx], curr_expr[right_parenthesis_idx + 1:])
        match = re.search('([^a-zA-Z0-9_]|^)le\(', curr_expr)
    return curr_expr


def extract_leading_quantifiers(ivy_expr):
    # return a 3-tuple (stripped ivy expr, universally quantified variable set, existentially quantified variable set)
    parts = ivy_expr.split('.')
    if len(parts) == 1:
        return ivy_expr, set(), set()
    first_part = parts[0]
    # extend the following line after supporting new modules
    if first_part.endswith('ring'):
        return ivy_expr, set(), set()
    quantifier_part, expr_part = parts
    quantifier_part, expr_part = quantifier_part.strip(), expr_part.strip()
    if quantifier_part.startswith('forall '):
        uqvars = quantifier_part[len('forall '):]
        uqvars = uqvars.split(',')
        uqvars = set([s.strip() for s in uqvars])
        return expr_part, uqvars, set()
    else:
        assert(quantifier_part.startswith('exists '))
        eqvars = quantifier_part[len('exists '):]
        eqvars = eqvars.split(',')
        eqvars = set([s.strip() for s in eqvars])
        return expr_part, set(), eqvars


def calc_quantified_expr(python_expr, qvars, qtype, tmp_var):
    # calculate the Python statements to calculate a quantified expression
    # e.g., python_expr = 'p(N) or q(N)', qvars = {'N': 'node'}, qtype = 'forall', tmp_var = 'tmp_var_6'
    # output ['tmp_var_6 = True', 'for N in range(node_num):', '\tif not (p(N) or q(N)):', '\t\ttmp_var_6 = False',
    #         '\t\tbreak']
    assert(qtype in ['forall', 'exists'])
    python_stmts = []
    if qtype == 'forall':
        python_stmts.append('{} = True'.format(tmp_var))
    else:
        python_stmts.append('{} = False'.format(tmp_var))
    indent_prefix = ''
    for qvar, type_name in qvars.items():
        for_stmt = '{}for {} in range({}_num):'.format(indent_prefix, qvar, type_name)
        indent_prefix += '\t'
        python_stmts.append(for_stmt)
    if qtype == 'forall':
        if_stmt = '{}if not ({}):'.format(indent_prefix, python_expr)
        indent_prefix += '\t'
        assign_stmt = '{}{} = False'.format(indent_prefix, tmp_var)
    else:
        if_stmt = '{}if ({}):'.format(indent_prefix, python_expr)
        indent_prefix += '\t'
        assign_stmt = '{}{} = True'.format(indent_prefix, tmp_var)
    break_stmt = '{}break'.format(indent_prefix)
    python_stmts.extend([if_stmt, assign_stmt, break_stmt])
    return python_stmts


def find_module(module_str):
    module_str = module_str.split('(')[0].strip()
    known_module = ['ring_topology', 'total_order']
    return module_str in known_module


def find_instantiated_module(instance_str):
    known_instance = ['ring', 'le']
    return instance_str in known_instance


def get_ring_initialization_block(element_type):
    element_size = '{}_num'.format(element_type)
    lines = ['# build ring topology',
             'btw = np.zeros(({}, {}, {}), dtype=bool)'.format(element_size, element_size, element_size),
             'for xx in range({}):'.format(element_size),
             '\tfor yy in range({}):'.format(element_size),
             '\t\tfor zz in range({}):'.format(element_size),
             '\t\t\tif xx != yy and xx != zz and yy != zz:',
             '\t\t\t\tbtw[xx, yy, zz] = (xx < yy < zz) | (zz < xx < yy) | (yy < zz < xx)']
    return lines


def get_python_header():
    return ['import numpy as np',
            'from collections import defaultdict',
            'from scipy.special import comb',
            'import time', 'import pandas as pd',
            'from itertools import product, permutations',
            '',
            'rng = np.random.default_rng(0)',
            'bool_num = 2']


def get_select_and_execute_python_block():
    lines = ['rng.shuffle(action_pool)',
             'action_selected, args_selected = None, None',
             'for action in action_pool:',
             '\trng.shuffle(argument_pool[action])',
             '\targument_candidates = argument_pool[action]',
             '\tfor args_candidate in argument_candidates:',
             "\t\tif func_from_name[action + '_prec'](*args_candidate):",
             '\t\t\taction_selected, args_selected = action, args_candidate',
             '\t\t\tbreak',
             '\tif action_selected is not None:',
             '\t\tbreak',
             'if action_selected is None:',
             '\t# action pool exhausted, start a new simulation',
             '\tbreak',
             'func_from_name[action_selected](*args_selected)']
    return lines
