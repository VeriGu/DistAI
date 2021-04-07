# this file parses an Ivy program and emits a protocol simulation script in Python
# a principled solution, as adopted by I4, is to call the Ivy compiler to generate the syntax tree
# however, since Ivy is not actively maintained, we ran into painful conflicting dependency issues
# so we eventually took an unorthodox approach: parse an Ivy file from scratch and translates it into Python
# this file does not aim to be a complete compiler, yet is enough to support the 14 protocols evaluated

# usage: python translate.py PROTOCOL_NAME NUM_


import sys
from translate_helper import *
from ivy_parser import *
from collections import Counter, defaultdict
from itertools import product, chain
import getopt

SAFETY_PROPERTY_ID = 1000000
DEFAULT_MAX_ITER = 50
SUBSAMPLE_PER_SAMPLE = 3
MAX_LITIRAL_INIT = 3
MAX_SIMULATE_ROUND = 1000
MAX_SIMULATE_ROUND_WIHOUT_PROGRESS = 20
DEFAULT_AXIOM_MAX_RETRY = 10


EXACT_NUM_OF_SAMPLE = [-1]  # only used by draw_tradeoff.py for convenience of terminating after a given number of samples


types = dict()  # key: type_name, value: minimum size
user_specified_min_size = {}  # key: type_name, value: minimum size
type_abbrs = dict()  # key: type_name, value: abbreviation (e.g., 'TR')
total_ordered_types = []  # element: type_name
relations = dict()  # key: relation_name, value: [type_name_1,...,type_name_k]
functions = dict()  # key: function_name, value: ([type_name_1,...,type_name_k], type_name)
individuals = dict()  # key: constant_name, value: type_name
instantiations = dict()  # key: module_name, value: [instance_1,...,instance_k]
module_relations = dict()  # key: relation_name, value: (individual_name, [type_name_1,...,type_name_k])
init_block = []     # each element is a line of Python code
actions = dict()  # key: action_name, value: [(param_name_1, type_name_1),...(param_name_k, type_name_k)]
action_precs = dict()  # key: action_name, value: a Python function in form of list of strings
action_trans = dict()  # key: action_name, value: a Python function in form of list of strings
action_prefixes = dict()  # key: action_name, value: a list of Python statements
axioms = dict()  # key: axiom_type, value: relation/function involved
invariants = []  # element: a string representing an invariant (safety property)
axiom_default_relations = []  # element: relation_name involved in axioms handled by default
python_codes = []  # each element is a line of Python code, will be written to .py file
tmp_var_counter = [1]  # used to name temp variables
bottleneck = [0]  # whether sampling is going to be bottleneck of system
vars_each_type = dict()  # key: type_name, value: [var_name_1,...,var_name_k]
predicate_columns = []  # element: a predicate (e.g., requested[N1, N2])
safety_relations = set()  # element: relation_name (when filled)


def parse_type(type_name):
    if len(type_name.split()) > 1:
        print('Invalid type name: {}'.format(type_name))
        exit(-1)
    if type_name in types:
        print('Redeclared type: {}'.format(type_name))
        exit(-1)
    types[type_name] = user_specified_min_size[type_name] if type_name in user_specified_min_size else 0


def parse_relation(relation_str):
    relation_parts = relation_str.replace(')', '(').split('(')
    if len(relation_parts) != 3:
        print('Unparsable relation: {}'.format(relation_str))
        exit(-1)
    relation_name, parameters_str, _ = relation_parts
    if len(relation_name.split()) > 1:
        print('Invalid relation name: {}'.format(relation_name))
        exit(-1)
    if relation_name in relations:
        print('Redeclared relation: {}'.format(relation_name))
        exit(-1)
    parameter_strs = parameters_str.split(',')
    relation_signature = []
    for parameter_str in parameter_strs:
        parameter_str_splitted = parameter_str.strip().split(':')
        if len(parameter_str_splitted) != 2:
            print('Relation {} has invalid parameter {}. Should be in the form var:type.'.format(relation_name,
                                                                                                 parameter_str))
            exit(-1)
        var_name, var_type = parameter_str_splitted
        var_type = var_type.strip()
        if var_type not in types:
            print('Undeclared type {} in relation {}'.format(var_type, relation_name))
            exit(-1)
        relation_signature.append(var_type)
    relations[relation_name] = relation_signature


def parse_function(function_str):
    function_str_splitted = function_str.replace(')', '(').split('(')
    if len(function_str_splitted) != 3 or not function_str_splitted[2].strip().startswith(':'):
        print('Invalid function declaration {}'.format(function_str))
        exit(-1)
    func_name, parameters_str, ret_type = function_str_splitted
    func_name, parameters_str, ret_type = func_name.strip(), parameters_str.strip(), ret_type.strip()[1:].strip()
    parameter_strs = parameters_str.split(',')
    func_signature = []
    for parameter_str in parameter_strs:
        parameter_str_splitted = parameter_str.strip().split(':')
        if len(parameter_str_splitted) != 2:
            print('Function {} has invalid parameter {}. Should be in the form var:type.'.format(func_name,
                                                                                                 parameter_str))
            exit(-1)
        var_name, var_type = parameter_str_splitted
        var_type = var_type.strip()
        if var_type not in types:
            print('Undeclared type {} in function {}'.format(var_type, func_name))
            exit(-1)
        func_signature.append(var_type)
    if ret_type not in types:
        print('Undeclared return type {} in function {}'.format(ret_type, func_name))
        exit(-1)
    functions[func_name] = (func_signature, ret_type)


def parse_individual(individual_str):
    individual_str_splitted = individual_str.split(':')
    if len(individual_str_splitted) != 2:
        print('Invalid individual declaration {}'.format(individual_str))
        exit(-1)
    idv_name, idv_type = individual_str_splitted
    idv_name, idv_type = idv_name.strip(), idv_type.strip()
    assert(idv_type in types or idv_type == 'bool')
    # consider individual abort_flag: bool
    individuals[idv_name] = idv_type


def get_partial_function_init_lines(relation_name, first_type, second_type, domain_is_first):
    lines = []
    lines.append('{} = np.zeros(({}_num, {}_num), dtype=bool)'.format(relation_name, first_type, second_type))
    if domain_is_first:
        lines.append('{}_f = rng.integers(0, {}_num, size=({}_num))'.format(relation_name, second_type, first_type))
        lines.append('for i in range({}_num):'.format(first_type))
        lines.append('\t{}[i, {}_f[i]] = True'.format(relation_name, relation_name))
    else:
        lines.append('{}_f = rng.integers(0, {}_num, size=({}_num))'.format(relation_name, first_type, second_type))
        lines.append('for i in range({}_num):'.format(second_type))
        lines.append('\t{}[{}_f[i], i] = True'.format(relation_name, relation_name))
    return lines


def default_axiom_rejection_sampling(axiom_str):
    bottleneck[0] = 3
    lines = []
    lines.append('# the following code block applies rejection sampling to generate predicates that satisfy axiom:')
    lines.append('# ' + axiom_str)
    lines.append('# you may consider manually improving its efficiency')
    tree_root = tree_parse_ivy_expr(axiom_str, None)
    tree_root = standardize_tree(tree_root)
    tree_node_list = all_nodes_of_tree(tree_root)
    relations_involved = []
    for tree_node in tree_node_list:
        if tree_node.node_type == 'predicate' and tree_node.metadata not in relations_involved:
            relations_involved.append(tree_node.metadata)
    python_bexpr, python_stmts = ivy_bexpr_to_python_bexpr(axiom_str)
    lines.append('predicates_valid = False')
    lines.append('for retry in range({}):'.format(DEFAULT_AXIOM_MAX_RETRY))
    indent_prefix = '\t'
    init_lines = []
    for relation_name in relations_involved:
        if 'partial-func' in axioms and relation_name in axioms['partial-func']:
            domain_is_first, is_invariant = axioms['partial-func'][relation_name]
            assert(is_invariant)
            first_type, second_type = relations[relation_name]
            init_lines.extend(get_partial_function_init_lines(relation_name, first_type, second_type, domain_is_first))
        else:
            var_num_list = []
            for type_name in relations[relation_name]:
                var_num_list.append(type_name + '_num')
            line = '{} = rng.integers(0, 2, size=({}), dtype=bool)'.format(relation_name, ', '.join(var_num_list))
            init_lines.append(line)
    lines.extend([(indent_prefix + s) for s in init_lines])
    lines.extend([(indent_prefix + s) for s in python_stmts])
    lines.append('{}if ({}):'.format(indent_prefix, python_bexpr))
    indent_prefix += '\t'
    lines.append('{}predicates_valid = True'.format(indent_prefix))
    lines.append('{}break'.format(indent_prefix))
    lines.append('if not predicates_valid:')
    lines.append('\tcontinue')
    lines.append('')
    return lines


def build_initialization_block():
    lines = []
    for relation in relations:
        if 'partial-func' not in axioms or relation not in axioms['partial-func']:
            var_num_list = []
            for type_name in relations[relation]:
                var_num_list.append(type_name + '_num')
            line = '{} = rng.integers(0, 2, size=({}), dtype=bool)'.format(relation, ', '.join(var_num_list))
            lines.append(line)
    for idv_name, idv_type in individuals.items():
        if idv_type == 'bool':
            lines.append('{} = rng.integers(0, 2, size=(1), dtype=bool)'.format(idv_name))
        else:
            assert(idv_type in types)
            lines.append('{} = rng.integers(0, {}_num)'.format(idv_name, idv_type))
    for func_name, (in_type_list, out_type) in functions.items():
        if 'one-to-one-f' in axioms and axioms['one-to-one-f'] == func_name:
            assert(len(in_type_list) == 1)
            in_type = in_type_list[0]
            lines.append('{} = rng.permutation({}_num)'.format(func_name, in_type))
        else:
            var_num_list = []
            for type_name in functions[func_name][0]:
                var_num_list.append(type_name + '_num')
            line = '{} = rng.integers(0, {}_num, size=({}))'.format(func_name, out_type, ', '.join(var_num_list))
            lines.append(line)
    for relation_name, (idv_name, type_list) in module_relations.items():
        if idv_name == 'ring':
            assert(relation_name == 'btw')
            assert(len(type_list) == 3 and type_list[0] == type_list[1] == type_list[2] and type_list[0] in types)
            element_type = type_list[0]
            lines.extend(get_ring_initialization_block(element_type))
    # in the future, we should support a relation with one relevant axiom handled explicitly and another axiom handled by default
    if 'reflexivity' in axioms:
        for relation_name, is_reflexive in axioms['reflexivity']:
            type_name = relations[relation_name][0]
            lines.extend(['for i in range({}_num):'.format(type_name), '\t{}[i,i] = {}'.format(relation_name, is_reflexive)])
    if 'symmetry' in axioms:
        for relation_name, is_symmetric in axioms['symmetry']:
            type_name = relations[relation_name][0]
            if is_symmetric:
                lines.extend(['for i in range(1, {}_num):'.format(type_name), '\tfor j in range(0, i):', '\t\t{}[i,j] = {}[j,i]'.format(relation_name, relation_name)])
            else:
                lines.extend(['for i in range(1, {}_num):'.format(type_name), '\tfor j in range(0, i):', '\t\t{}[i,j] = not {}[j,i]'.format(relation_name, relation_name)])
                lines.extend(['for i in range({}_num):'.format(type_name), '\t{}[i,i] = False'.format(relation_name)])
    if 'least' in axioms:
        lines.append('{} = 0'.format(axioms['least']))
    if 'nonleast' in axioms:
        for nonleast_idv in axioms['nonleast']:
            lines.append('{} = rng.integers(1, {}_num)'.format(nonleast_idv, individuals[nonleast_idv]))
    if 'nequal' in axioms:
        for const1, const2 in axioms['nequal']:
            assert(const1 in individuals and const2 in individuals and individuals[const1] == individuals[const2])
            lines.append('{}, {} = rng.choice({}_num, 2, replace=False)'.format(const1, const2, individuals[const1]))
    if 'qmembership' in axioms:
        for qmember_relation in axioms['qmembership']:
            assert(len(relations[qmember_relation]) == 2)
            mtype, qtype = relations[qmember_relation]
            lines.append('{} = np.zeros(({}_num, {}_num), dtype=bool)'.format(qmember_relation, mtype, qtype))
            lines.append('for q in range({}_num):'.format(qtype))
            lines.append('\tqsize = rng.integers(np.ceil({}_num/2), {}_num + 1)'.format(mtype, mtype))
            lines.append('\ttmp_members = rng.choice(list(range({}_num)), qsize, replace=False)'.format(mtype))
            lines.append('\tfor m in tmp_members:')
            lines.append('\t\t{}[m,q] = True'.format(qmember_relation))
    if 'default' in axioms:
        for default_axiom in axioms['default']:
            lines.extend(default_axiom_rejection_sampling(default_axiom))
    if 'partial-func' in axioms:
        noninvariant_partial_funcs = []
        for relation_name, (domain_is_first, is_invariant) in axioms['partial-func'].items():
            if relation_name not in axiom_default_relations:
                first_type, second_type = relations[relation_name]
                if is_invariant:
                    lines.extend(get_partial_function_init_lines(relation_name, first_type, second_type, domain_is_first))
                else:
                    lines.extend(get_partial_function_init_lines(relation_name, first_type, second_type, domain_is_first))
                    noninvariant_partial_funcs.append(relation_name)
        for relation_name in noninvariant_partial_funcs:
            del(axioms['partial-func'][relation_name])
    if 'cond-total-order' in axioms:
        for relation1, relation2 in axioms['cond-total-order']:
            type2 = relations[relation2][0]
            lines.append('')
            lines.append('# conditional total order')
            lines.append('{} = np.zeros(({}_num, {}_num), dtype=bool)'.format(relation2, type2, type2))
            lines.append('for X in range({}_num):'.format(type2))
            lines.append('\tfor Y in range({}_num):'.format(type2))
            lines.append('\t\t{}[X, Y] = ({}_f[X] == {}_f[Y] and X <= Y)'.format(relation2, relation1, relation1))
    lines.append('')
    return lines


def ivy_pure_expr_to_python_pure_expr(ivy_expr):
    assert(type(ivy_expr) == str)
    assert('->' not in ivy_expr)
    ivy_expr = ivy_expr.strip()
    expr_as_list = list(ivy_expr)
    qvars = dict()   # a dict from quantified variable name to its type
    strs_to_remove = []  # e.g., 'ring' in 'ring.btw'
    for name in list(relations.keys()) + list(functions.keys()) + list(module_relations.keys()):
        if name in relations:
            type_list = relations[name]
        elif name in module_relations:
            strs_to_remove.append(module_relations[name][0] + '.')
            type_list = module_relations[name][1]
        else:
            type_list = functions[name][0]
        matches = re.finditer('([^0-9a-zA-Z_]|^)' + name + '\(', ivy_expr)
        for match in matches:
            # replace () in Ivy with [] in Python
            left_parenthesis = match.end() - 1
            right_parenthesis = find_closing_parenthesis(ivy_expr, left_parenthesis)
            expr_as_list[left_parenthesis] = '['
            expr_as_list[right_parenthesis] = ']'
            parameters_str = ivy_expr[left_parenthesis + 1: right_parenthesis]
            parameter_strs = parameters_str.split(',')
            assert(len(parameter_strs) == len(type_list))
            for i, param in enumerate(parameter_strs):
                param = param.strip()
                if 'A' <= param[0] <= 'Z':
                    # quantified variable found
                    type_name = type_list[i]
                    qvars[param] = type_name
    python_expr = ''.join(expr_as_list)
    for str_to_remove in strs_to_remove:
        python_expr = python_expr.replace(str_to_remove, '')
    syntax_map = {'&': 'and', '|': 'or', '~=': '!=', '=': '==', '~': 'not ', '<=': '<=', 'true': 'True', 'false': 'False'}
    regex = re.compile("|".join(map(re.escape, syntax_map.keys())))
    python_expr = regex.sub(lambda match: syntax_map[match.group(0)], python_expr)
    return python_expr, qvars


def ivy_bexpr_to_python_bexpr(ivy_bexpr):
    # return a 2-tuple with type (python_bexpr, python_stmts)
    # python_bexpr: a string, the equivalent Python expression
    # python_stmts: the Python statements needed to define python_bexpr
    ivy_bexpr = remove_implication(ivy_bexpr)
    ivy_bexpr = translate_remove_le(ivy_bexpr)
    sub_ivy_bexprs = separate_ivy_bexpr_by_and(ivy_bexpr)
    pure_python_exprs, python_stmt_blocks, tmp_vars = [], [], []
    for sub_ivy_bexpr in sub_ivy_bexprs:
        sub_ivy_bexpr = strip_parenthesis(sub_ivy_bexpr)
        pure_expr, uqvars, eqvars = extract_leading_quantifiers(sub_ivy_bexpr)
        python_expr, qvars = ivy_pure_expr_to_python_pure_expr(pure_expr)
        if len(qvars) == 0:
            pure_python_exprs.append('({})'.format(python_expr))
        else:
            tmp_var = 'tmp_var_{}'.format(tmp_var_counter[0])
            tmp_var_counter[0] += 1
            tmp_vars.append(tmp_var)
            if len(eqvars) > 0:
                for qvar in qvars:
                    assert(qvar in eqvars)
                python_block = calc_quantified_expr(python_expr, qvars, 'exists', tmp_var)
            else:
                python_block = calc_quantified_expr(python_expr, qvars, 'forall', tmp_var)
            python_stmt_blocks.append(python_block)
    anded_pure_expr = ' and '.join(pure_python_exprs)
    if len(python_stmt_blocks) == 0:
        python_bexpr = anded_pure_expr
        python_stmts = []
    else:
        python_bexpr = ' and '.join(tmp_vars)
        python_stmts = []
        indent_prefix = ''
        if len(pure_python_exprs) > 0:
            python_stmts.append('{} = False'.format(' = '.join(tmp_vars)))
            python_stmts.append('if {}:'.format(anded_pure_expr))
            indent_prefix += '\t'
        for i, python_block in enumerate(python_stmt_blocks):
            python_stmts.extend([(indent_prefix + s) for s in python_block])
            if i != len(python_stmt_blocks) - 1:
                python_stmts.append('{}if {}:'.format(indent_prefix, tmp_vars[i]))
            indent_prefix += '\t'
    return python_bexpr, python_stmts


def translate_assignment(lvalue, rvalue):
    lvalue, rvalue = lvalue.strip(), rvalue.strip()
    rvalue = translate_remove_le(rvalue)
    lexpr, lqvars = ivy_pure_expr_to_python_pure_expr(lvalue)
    rexpr, rqvars = ivy_pure_expr_to_python_pure_expr(rvalue)
    for qvar, type_name in rqvars.items():
        assert(lqvars[qvar] == type_name)
    if lvalue in individuals and individuals[lvalue] == 'bool':
        lexpr = '{}[0]'.format(lvalue)
    if rvalue == '*':  # any value:
        rexpr = 'rng.integers(0, 2, dtype=bool)'
    python_stmts = []
    if len(lqvars) == 0:
        assign_line = '{} = {}'.format(lexpr, rexpr)
        python_stmts.append(assign_line)
    else:
        indent_prefix = ''
        for i, (qvar, type_name) in enumerate(lqvars.items()):
            for_line = '{}for {} in range({}_num):'.format(indent_prefix, qvar, type_name)
            python_stmts.append(for_line)
            indent_prefix += '\t'
        assign_line = '{}{} = {}'.format(indent_prefix, lexpr, rexpr)
        python_stmts.append(assign_line)
    return python_stmts


def parse_init_stmt(init_stmt):
    assert(init_stmt[-1] == ';')
    init_stmt_splitted = init_stmt[:-1].split(':=')
    if len(init_stmt_splitted) != 2:
        if init_stmt.startswith('require'):
            # non-deterministic initialization
            axiom_str = init_stmt[len('require') + 1: -1].strip()
            parse_axiom(axiom_str, from_init_not_axiom=True)
            return []
        else:
            print('Ill-formed initialization statement: {}'.format(init_stmt))
            exit(-1)
    lvalue, rvalue = init_stmt_splitted
    python_stmts = translate_assignment(lvalue, rvalue)
    return python_stmts


def parse_require_stmt(require_stmt):
    # called by parse_action, parse one require line at the beginning of an action
    python_bexpr, python_stmts = ivy_bexpr_to_python_bexpr(require_stmt)
    first_line = 'if not ({}):'.format(python_bexpr)
    second_line = '\treturn False'
    return python_stmts + [first_line, second_line]


def parse_instantiation(instantiate_str):
    colon_splitted = instantiate_str.split(':')
    assert (len(colon_splitted) in [1, 2])
    var_name = ''
    if len(colon_splitted) == 1:
        parts = colon_splitted[0].replace(')', '(').split('(')
    else:
        var_name, inst = colon_splitted[0].strip(), colon_splitted[1].strip()
        parts = inst.replace(')', '(').split('(')
    assert (len(parts) == 3)
    module_name, instance_name, _ = parts
    module_name, instance_name = module_name.strip(), instance_name.strip()
    if module_name not in instantiations:
        instantiations[module_name] = []
    if len(colon_splitted) == 1:
        instantiations[module_name].append(instance_name)
    else:
        instantiations[module_name].append((instance_name, var_name))
        # handle ring.btw
        if module_name == 'ring_topology':
            assert(var_name == 'ring')
            module_relations['btw'] = ('ring', [instance_name, instance_name, instance_name])


def parse_axiom(axiom_str, from_init_not_axiom=False):
    axiom_str = axiom_str.strip()
    # print('parsing axiom {}'.format(axiom_str))
    # now we consider two type of axioms p(X,Y1) /\ p(X,Y2) -> Y1 = Y2 for relation p
    # and f(X) = f(Y) -> X = Y for function f
    axiom_recognized = False
    tree_root = tree_parse_ivy_expr(axiom_str, None)
    tree_root = standardize_tree(tree_root)
    if tree_root.node_type == 'imply':
        left_child, right_child = tree_root.children
        if right_child.node_type == 'equal' and right_child.children[0].node_type == 'qvar' and right_child.children[1].node_type == 'qvar' and right_child.children[0].substr != right_child.children[1].substr:
            rqvar1, rqvar2 = right_child.children[0].substr, right_child.children[1].substr
            if left_child.node_type == 'equal':
                lclc, lcrc = left_child.children
                if lclc.node_type == lcrc.node_type == 'predicate' and len(lclc.children) == len(lcrc.children) == 1 and lclc.metadata in functions and {lclc.children[0].substr, lcrc.children[0].substr} == {rqvar1, rqvar2}:
                    axiom_recognized = True
                    func_name = lclc.metadata
                    # currently only supports one bijection function
                    assert ('one-to-one-f' not in axioms)
                    axioms['one-to-one-f'] = func_name
            elif left_child.node_type == 'and' and len(left_child.children) == 2:
                llc, lrc = left_child.children
                if llc.node_type == lrc.node_type == 'predicate' and llc.metadata == lrc.metadata and len(llc.children) == len(lrc.children) == 2:
                    relation_name = llc.metadata
                    (var1, var2), (var3, var4) = llc.children, lrc.children
                    if var1.node_type == var2.node_type == var3.node_type == var4.node_type == 'qvar':
                        var1, var2, var3, var4 = var1.substr, var2.substr, var3.substr, var4.substr
                        if ({var1, var3} == {rqvar1, rqvar2} and var2 == var4) or ({var2, var4} == {rqvar1, rqvar2} and var1 == var3):
                            # binary relation is partial function
                            axiom_recognized = True
                            if 'partial-func' not in axioms:
                                axioms['partial-func'] = dict()
                            domain_is_first = {var2, var4} == {rqvar1, rqvar2} and var1 == var3
                            axioms['partial-func'][relation_name] = (domain_is_first, not from_init_not_axiom)
                        elif var1 == var4 and var2 == var3:
                            axiom_recognized = True
                            # anti-symmetry
                            if 'symmetry' not in axioms:
                                axioms['symmetry'] = []
                            axioms['symmetry'].append((relation_name, False))
            elif left_child.node_type == 'and' and len(left_child.children) == 3:
                lc1, lc2, lc3 = left_child.children
                if lc1.node_type == lc2.node_type == lc3.node_type == 'predicate' and len({lc1.metadata, lc2.metadata, lc3.metadata}) == 3 and len(lc1.children) == len(lc2.children) == len(lc3.children) == 2:
                    (lc1c1, lc1c2), (lc2c1, lc2c2), (lc3c1, lc3c2) = lc1.children, lc2.children, lc3.children
                    if lc1c1.node_type == lc1c2.node_type == lc2c1.node_type == lc2c2.node_type == lc3c1.node_type == lc3c2.node_type == 'qvar':
                        lc1c1, lc1c2, lc2c1, lc2c2, lc3c1, lc3c2 = lc1c1.substr, lc1c2.substr, lc2c1.substr, lc2c2.substr, lc3c1.substr, lc3c2.substr
                        if len({lc1c1, lc2c1, lc3c1}) == 3 and len({lc1c2, lc2c2, lc3c2}) == 2:
                            lc1c1, lc1c2, lc2c1, lc2c2, lc3c1, lc3c2 = lc1c2, lc1c1, lc2c2, lc2c1, lc3c2, lc3c1
                        if len({lc1c1, lc2c1, lc3c1}) == 2 and len({lc1c2, lc2c2, lc3c2}) == 3:
                            if lc1c1 == lc3c1:
                                lc2c1, lc2c2, lc3c1, lc3c2 = lc3c1, lc3c2, lc2c1, lc2c2
                            elif lc2c1 == lc3c1:
                                lc1c1, lc1c2, lc3c1, lc3c2 = lc3c1, lc3c2, lc1c1, lc1c2
                            if lc2c2 == lc3c1:
                                lc1c1, lc1c2, lc2c1, lc2c2 = lc2c1, lc2c2, lc1c1, lc1c2
                            if lc1c2 == lc3c1 and {lc2c2, lc3c2} == {rqvar1, rqvar2}:
                                # locality
                                axiom_recognized = True
                                pass
        elif right_child.node_type == 'predicate' and len(right_child.children) == 2 and right_child.children[0].node_type == right_child.children[1].node_type == 'qvar' and right_child.children[0].substr != right_child.children[1].substr:
            rqvar1, rqvar2 = right_child.children[0].substr, right_child.children[1].substr
            if left_child.node_type == 'and' and len(left_child.children) == 2:
                llc, lrc = left_child.children
                if llc.node_type == lrc.node_type == 'predicate' and llc.metadata == lrc.metadata and len(llc.children) == len(lrc.children) == 2:
                    relation_name = llc.metadata
                    (var1, var2), (var3, var4) = llc.children, lrc.children
                    if var1.node_type == var2.node_type == var3.node_type == var4.node_type == 'qvar':
                        var1, var2, var3, var4 = var1.substr, var2.substr, var3.substr, var4.substr
                        if len({var1, var2, var3, var4}) == 3 and ((var2 == var3 and {var1, var4} == {rqvar1, rqvar2}) or (var1 == var4 and {var2, var3} == {rqvar1, rqvar2})):
                            axiom_recognized = True
                            if 'transitivity' not in axioms:
                                axioms['transitivity'] = []
                            axioms['transitivity'].append(relation_name)
                elif (llc.node_type == 'predicate' and lrc.node_type == 'or') or (llc.node_type == 'or' and lrc.node_type == 'predicate'):
                    if llc.node_type == 'or':
                        llc, lrc = lrc, llc
                    if len(llc.children) == 2 and llc.metadata == right_child.metadata:
                        relation1 = llc.metadata
                        if len(lrc.children) == 2 and lrc.children[0].node_type == lrc.children[1].node_type == 'predicate' and lrc.children[0].metadata == lrc.children[1].metadata and len(lrc.children[0].children) == len(lrc.children[1].children) == 2:
                            relation2 = lrc.children[0].metadata
                            (var1, var2), (var3, var4), (var5, var6) = llc.children, lrc.children[0].children, lrc.children[1].children
                            if var1.node_type == var2.node_type == var3.node_type == var4.node_type == var5.node_type == var6.node_type == 'qvar':
                                var1, var2, var3, var4, var5, var6 = var1.substr, var2.substr, var3.substr, var4.substr, var5.substr, var6.substr
                                if var3 == var6 and var4 == var5 and var3 != var4 and {var2, rqvar2} == {var3, var4} and var1 == rqvar1:
                                    # conditional closed
                                    axiom_recognized = True
                                    if 'cond-closed' not in axioms:
                                        axioms['cond-closed'] = []
                                    axioms['cond-closed'].append((relation1, relation2))
        elif right_child.node_type == 'or':
            if len(right_child.children) == 2 and right_child.children[0].node_type == right_child.children[1].node_type == 'predicate':
                rlc, rrc = right_child.children
                if rlc.metadata == rrc.metadata and len(rlc.children) == len(rrc.children) == 2 and rlc.children[0].node_type == rlc.children[1].node_type == rrc.children[0].node_type == rrc.children[1].node_type == 'qvar':
                    relation2 = rlc.metadata
                    rqvar1, rqvar2, rqvar3, rqvar4 = rlc.children[0].substr, rlc.children[1].substr, rrc.children[0].substr, rrc.children[1].substr
                    if left_child.node_type == 'and' and len(left_child.children) == 2 and left_child.children[0].node_type == left_child.children[1].node_type == 'predicate':
                        llc, lrc = left_child.children
                        if llc.metadata == lrc.metadata and len(llc.children) == len(lrc.children) == 2 and llc.children[0].node_type == llc.children[1].node_type == lrc.children[0].node_type == lrc.children[1].node_type == 'qvar':
                            relation1 = llc.metadata
                            lqvar1, lqvar2, lqvar3, lqvar4 = llc.children[0].substr, llc.children[1].substr, lrc.children[0].substr, lrc.children[1].substr
                            if lqvar1 == lqvar3 and rqvar1 == rqvar4 and rqvar2 == rqvar3 and rqvar1 != rqvar2 and {lqvar2, lqvar4} == {rqvar1, rqvar2}:
                                # conditional total
                                axiom_recognized = True
                                if 'cond-total' not in axioms:
                                    axioms['cond-total'] = []
                                axioms['cond-total'].append((relation1, relation2))
    elif tree_root.node_type == 'or':
        if len(tree_root.children) == 2:
            left_child, right_child = tree_root.children
            if left_child.node_type == right_child.node_type == 'not':
                lcc, rcc = left_child.children[0], right_child.children[0]
                if lcc.node_type == rcc.node_type == 'predicate' and len(lcc.children) == len(rcc.children) == 2 and lcc.children[0].substr != lcc.children[1].substr and lcc.children[0].substr == rcc.children[1].substr and lcc.children[1].substr == rcc.children[0].substr:
                    axiom_recognized = True
                    if 'symmetry' not in axioms:
                        axioms['symmetry'] = []
                    axioms['symmetry'].append((lcc.metadata, False))
            elif left_child.node_type == 'not' or right_child.node_type == 'not':
                lcc = left_child.children[0] if left_child.node_type == 'not' else left_child
                rcc = right_child.children[0] if right_child.node_type == 'not' else right_child
                if lcc.node_type == rcc.node_type == 'predicate' and len(lcc.children) == len(rcc.children) == 2 and lcc.children[0].substr != lcc.children[1].substr and lcc.children[0].substr == rcc.children[1].substr and lcc.children[1].substr == rcc.children[0].substr:
                    axiom_recognized = True
                    if 'symmetry' not in axioms:
                        axioms['symmetry'] = []
                    axioms['symmetry'].append((lcc.metadata, True))
    elif tree_root.node_type == 'predicate':
        if len(tree_root.children) == 2:
            if tree_root.children[0].substr == tree_root.children[1].substr:
                axiom_recognized = True
                if 'reflexivity' not in axioms:
                    axioms['reflexivity'] = []
                axioms['reflexivity'].append((tree_root.metadata, True))
            elif tree_root.metadata == 'le' and tree_root.children[0].node_type == 'const' and tree_root.children[1].node_type == 'qvar':
                axiom_recognized = True
                assert('least' not in axioms)  # one total ordered set cannot have two least elements
                axioms['least'] = tree_root.children[0].substr
    elif tree_root.node_type == 'not':
        child_node = tree_root.children[0]
        if child_node.node_type == 'predicate':
            if len(child_node.children) == 2 and child_node.children[0].substr == child_node.children[1].substr:
                axiom_recognized = True
                if 'reflexivity' not in axioms:
                    axioms['reflexivity'] = []
                axioms['reflexivity'].append((child_node.metadata, False))
    elif tree_root.node_type == 'nequal':
        left_child, right_child = tree_root.children
        if left_child.node_type == 'const' and right_child.node_type == 'const' and left_child.substr != right_child.substr:
            if 'least' in axioms and (left_child.substr == axioms['least'] or right_child.substr == axioms['least']):
                axiom_recognized = True
                nonleast_idv = right_child.substr if left_child.substr == axioms['least'] else left_child.substr
                if 'nonleast' not in axioms:
                    axioms['nonleast'] = []
                axioms['nonleast'].append(nonleast_idv)
            else:
                axiom_recognized = True
                const1, const2 = left_child.substr, right_child.substr
                if 'nequal' not in axioms:
                    axioms['nequal'] = []
                axioms['nequal'].append((const1, const2))
    elif tree_root.node_type == 'forall':
        uqvars = tree_root.metadata
        if len(uqvars) == 2 and tree_root.children[0].node_type == 'exists':
            uqvar1, uqvar2 = uqvars
            exists_node = tree_root.children[0]
            if len(exists_node.metadata) == 1 and exists_node.children[0].node_type == 'and' and len(exists_node.children[0].children) == 2:
                eqvar = exists_node.metadata[0]
                and_left, and_right = exists_node.children[0].children
                if and_left.node_type == and_right.node_type == 'predicate' and and_left.metadata == and_right.metadata and len(and_left.children) == len(and_right.children) == 2:
                    relation_name = and_left.metadata
                    var1, var2, var3, var4 = and_left.children[0].substr, and_left.children[1].substr, and_right.children[0].substr, and_right.children[1].substr
                    if var1 == var3 == eqvar and {uqvar1, uqvar2} == {var2, var4}:
                        # quorum set membership
                        axiom_recognized = True
                        if 'qmembership' not in axioms:
                            axioms['qmembership'] = []
                        axioms['qmembership'].append(relation_name)
                        bottleneck[0] = 1
    if not axiom_recognized:
        if 'default' not in axioms:
            axioms['default'] = []
        axioms['default'].append(axiom_str)
        tree_node_list = all_nodes_of_tree(tree_root)
        for tree_node in tree_node_list:
            if tree_node.node_type == 'predicate':
                axiom_default_relations.append(tree_node.metadata)


def parse_if_block(if_cond, if_stmts):
    lines = []
    python_bexpr, python_pre_stmts = ivy_bexpr_to_python_bexpr(if_cond)
    lines.extend(python_pre_stmts)
    lines.append('if {}:'.format(python_bexpr))
    body_lines = []
    for line in if_stmts:
        lvalue, rvalue = line[: -1].split(':=')
        python_assign_stmts = translate_assignment(lvalue, rvalue)
        body_lines.extend(python_assign_stmts)
    lines.extend([('\t' + s) for s in body_lines])
    return lines


def simplify_python_conds_and_action_params_when_this_require_is_partial_function(action_name, require_stmt, python_conds):
    if 'partial-func' not in axioms:
        return
    tree_root = tree_parse_ivy_expr(require_stmt, None)
    if tree_root.node_type == 'predicate' and len(tree_root.children) == 2:
        relation_name = tree_root.metadata
        left_child, right_child = tree_root.children
        if relation_name in axioms['partial-func'] and left_child.node_type == right_child.node_type == 'const':
            # indeed partial function
            domain_is_first, _ = axioms['partial-func'][relation_name]
            const1, const2 = left_child.substr, right_child.substr
            python_conds.clear()
            if domain_is_first:
                action_prefixes[action_name].append('{} = {}_f[{}]'.format(const2, relation_name, const1))
                param_to_remove = const2
            else:
                action_prefixes[action_name].append('{} = {}_f[{}]'.format(const1, relation_name, const2))
                param_to_remove = const1
            param_to_remove_index = -1
            for i, (param, type_name) in enumerate(actions[action_name]):
                if param == param_to_remove:
                    param_to_remove_index = i
                    break
            assert(param_to_remove_index >= 0)
            del(actions[action_name][param_to_remove_index])


def parse_assume_block(tainted_exprs, assume_stmts):
    if len(assume_stmts) == 0:
        return []
    tainted_relations = dict()
    for tainted_expr in tainted_exprs:
        expr_tree_root = tree_parse_ivy_expr(tainted_expr, None)
        assert(expr_tree_root.node_type == 'predicate')
        tainted_relations[expr_tree_root.metadata] = []
        for i, child in enumerate(expr_tree_root.children):
            if child.node_type == 'const':
                tainted_relations[expr_tree_root.metadata].append((i, child.substr))

    # one may extend this code block to natively support more types of non-deterministic patterns
    bottleneck[0] = 2
    explicit_handling = False
    templates_found = defaultdict(list)
    lines = []
    if len(tainted_relations) == 1:
        tainted_relation, const_indices = list(tainted_relations.keys())[0], list(tainted_relations.values())[0]
        for assumt_stmt in assume_stmts:
            tree_root = tree_parse_ivy_expr(assumt_stmt, None)
            tree_root = standardize_tree(tree_root)
            nonstandard = False
            if tree_root.node_type == 'imply':
                left_child, right_child = tree_root.children
                all_nodes_of_right_subtree = all_nodes_of_tree(right_child)
                for node1 in all_nodes_of_right_subtree:
                    if node1.node_type == 'predicate':
                        relation_name = node1.metadata
                        if relation_name != tainted_relation:
                            nonstandard = True
                            break
                        # check whether not randomized (constant) dimensions match
                        for (i, const_str) in const_indices:
                            if node1.children[i].node_type != 'const' or node1.children[i].substr != const_str:
                                nonstandard = True
                                break
                        if nonstandard:
                            break
                if nonstandard:
                    continue
                if left_child.node_type == 'predicate' and len(left_child.children) == 1 and right_child.node_type == 'not':
                    rcc = right_child.children[0]
                    if rcc.node_type == 'and' and len(rcc.children) == 3:
                        rcc1, rcc2, rcc3 = rcc.children
                        if rcc1.node_type == 'nequal' and rcc2.node_type == rcc3.node_type == 'predicate' and rcc2.metadata == rcc3.metadata == tainted_relation:
                            if {rcc1.children[0].substr, rcc1.children[1].substr} == {rcc2.children[1].substr, rcc3.children[1].substr} and rcc2.children[0].substr == rcc3.children[0].substr:
                                templates_found['cons1'].append(left_child.metadata)
                elif left_child.node_type == 'and' and len(left_child.children) == 2 and left_child.children[0].node_type == left_child.children[1].node_type == 'predicate' and left_child.children[0].metadata == left_child.children[1].metadata:
                    left_relation = left_child.children[0].metadata
                    if len(left_child.children[0].children) == len(left_child.children[1].children) == 1:
                        left1, left2 = left_child.children[0].children[0], left_child.children[1].children[0]
                        if right_child.node_type == 'or' and len(right_child.children) == 2 and right_child.children[0].node_type == right_child.children[1].node_type == 'and' and len(right_child.children[0].children) == len(right_child.children[1].children) == 2:
                            right1, right2, right3, right4 = right_child.children[0].children[0], right_child.children[0].children[1], right_child.children[1].children[0], right_child.children[1].children[1]
                            if right3.node_type == right4.node_type == 'not':
                                right3, right4 = right3.children[0], right4.children[0]
                                if right1.node_type == right2.node_type == right3.node_type == right4.node_type == 'predicate' and right1.metadata == right2.metadata == right3.metadata == right4.metadata == tainted_relation:
                                    if {left1.substr, left2.substr} == {right1.children[0].substr, right2.children[0].substr} == {right3.children[0].substr, right4.children[0].substr} and right1.children[1].node_type == right2.children[1].node_type == 'const' and right1.children[1].substr == right2.children[1].substr and right3.children[1].node_type == right4.children[1].node_type == 'qvar' and right3.children[1].substr == right4.children[1].substr:
                                        templates_found['cons2'].append((left_relation, right1.children[1].substr))
                        elif right_child.node_type == 'predicate' and left1.node_type == 'const' and left2.node_type == 'qvar':
                            right1, right2 = right_child.children[0], right_child.children[1]
                            if left2.substr == right1.substr and right2.node_type == 'const':
                                templates_found['cons3'].append((left_relation, left1.substr, right2.substr))
    if len(templates_found['cons1']) == len(templates_found['cons2']) == len(templates_found['cons3']) == 1 and len(templates_found.keys()) == 3:
        relation1, (relation2, const2), (relation3, const31, const32) = templates_found['cons1'][0], templates_found['cons2'][0], templates_found['cons3'][0]
        if relation1 == relation2 == relation3 and const2 == const32:
            explicit_handling = True
            lines.append('if {}[{}] or rng.random() > 0.5:'.format(relation1, const31))
            lines.append('\tfor tmp_loop_var in range({}_num):'.format(relations[relation1][0]))
            lines.append('\t\t{}[tmp_loop_var, {}, {}] = True'.format(tainted_relation, const2, ','.join([node.substr for node in expr_tree_root.children[2:]])))

    if not explicit_handling:
        lines.append('# Non-deterministic action handled by rejection sampling. This can be slow. Consider options 1 or 2')
        lines.append('# 1) support this type of blackbox behavior in translate.py')
        lines.append('# 2) modify the while loop below to make sampling more efficient')
        lines.append('while True:')
        for assume_stmt in assume_stmts:
            python_bexpr, python_stmts = ivy_bexpr_to_python_bexpr(assume_stmt)
            lines.extend([('\t' + s) for s in python_stmts])
            lines.append('\tif not {}:'.format(python_bexpr))
            for ivy_expr in tainted_exprs:
                assignment_lines = translate_assignment(ivy_expr, '*')
                lines.extend([('\t\t' + s) for s in assignment_lines])
            lines.append('\t\tcontinue')
        lines.append('\tbreak')
    return lines


def parse_action(action_str, action_buffer):
    assert(action_str.count('=') == 1)
    action_str, left_brace = action_str.split('=')
    assert(left_brace.strip() == '{')
    action_str_splitted = action_str.replace(')', '(').split('(')
    if len(action_str_splitted) == 1:
        action_name, parameters_str = action_str_splitted[0], ''
    elif len(action_str_splitted) == 3:
        action_name, parameters_str, _ = action_str_splitted
    else:
        print('Ill-formed action declaration {}. Should be action action_name(parameters) = {{'.format(action_str))
        exit(-1)
        return  # suppress warning only
    action_name, parameters_str = action_name.strip(), parameters_str.strip()
    assert((action_name not in actions) and (action_name not in action_precs) and (action_name not in action_trans))
    if action_name == 'test':
        # in our system, 'test' is reserved for an action used to represent the unsafe state
        for line in action_buffer:
            line = line.strip()
            if line.startswith('require') and line[-1] == ';':
                tree_root = tree_parse_ivy_expr(line[len('require') + 1: -1], None)
                all_nodes = all_nodes_of_tree(tree_root)
                for tree_node in all_nodes:
                    if tree_node.node_type == 'predicate':
                        safety_relations.add(tree_node.metadata)
        return
    actions[action_name], action_precs[action_name], action_trans[action_name], action_prefixes[action_name] = [], [], [], []
    if len(parameters_str) == 0:
        parameter_strs = []
    else:
        parameter_strs = parameters_str.split(',')
    for name_type_pair in parameter_strs:
        name_type_pair = name_type_pair.split(':')
        assert(len(name_type_pair) == 2)
        param, type_name = name_type_pair[0].strip(), name_type_pair[1].strip()
        actions[action_name].append((param, type_name))
    in_prec, in_if = True, False
    if_cond, if_stmts = '', []
    assume_stmts, tainted_exprs = [], []
    for line in action_buffer:
        assert (line[-1] == ';' or (line.startswith('if') and line[-1] == '{'))
        if (line.startswith('require') or line.startswith('assume')) and in_prec:
            start_str = 'require' if line.startswith('require') else 'assume'
            require_stmt = line[len(start_str)+1: -1].strip()
            python_conds = parse_require_stmt(require_stmt)
            simplify_python_conds_and_action_params_when_this_require_is_partial_function(action_name, require_stmt, python_conds)
            for python_cond in python_conds:
                action_precs[action_name].append('\t{}'.format(python_cond))
        else:
            assert(in_prec or not line.startswith('require'))
            in_prec = False
            if in_if:
                if line == '};':
                    python_stmts = parse_if_block(if_cond, if_stmts)
                    for python_stmt in python_stmts:
                        action_trans[action_name].append('\t{}'.format(python_stmt))
                    in_if = False
                else:
                    assert(not line.startswith('if'))  # no nested if now
                    if_stmts.append(line)
            elif line.startswith('if') and line[-1] == '{':
                if_cond = strip_parenthesis(line[2:-1].strip()).strip()
                in_if = True
            elif line.startswith('assume') and line[-1] == ';':
                assume_stmts.append(line[len('assume'): -1].strip())
            else:
                assign_splitted = line[: -1].split(':=')
                assert(len(assign_splitted) == 2)
                lvalue, rvalue = assign_splitted[0].strip(), assign_splitted[1].strip()
                if rvalue == '*':
                    tainted_exprs.append(lvalue)
                python_stmts = translate_assignment(lvalue, rvalue)
                for python_stmt in python_stmts:
                    action_trans[action_name].append('\t{}'.format(python_stmt))
    python_stmts = parse_assume_block(tainted_exprs, assume_stmts)
    for python_stmt in python_stmts:
        action_trans[action_name].append('\t{}'.format(python_stmt))
    action_precs[action_name].append('\treturn True')
    # finally, add function declaration and prefix lines (for partial functions) at top
    action_precs[action_name][:0] = [('\t' + s) for s in action_prefixes[action_name]]
    action_trans[action_name][:0] = [('\t' + s) for s in action_prefixes[action_name]]
    param_list = [param for (param, type_name) in actions[action_name]]
    cond_func_decl = 'def {}_prec({}):'.format(action_name, ', '.join(param_list))
    action_precs[action_name].insert(0, cond_func_decl)
    func_decl = 'def {}({}):'.format(action_name, ', '.join(param_list))
    action_trans[action_name].insert(0, func_decl)


def parse_invariant(invariant_lines):
    inv_str = ' '.join(invariant_lines).strip()
    assert(inv_str.startswith('[{}]'.format(SAFETY_PROPERTY_ID)))
    inv_str = inv_str[len('[{}]'.format(SAFETY_PROPERTY_ID)) + 1:].strip()
    tree_root = tree_parse_ivy_expr(inv_str, None)
    if tree_root.node_type == 'and':
        inv_str_list = [child.substr for child in tree_root.children]
    else:
        inv_str_list = [tree_root.substr]
    invariants.extend(inv_str_list)


def parse_ivy_file(ivy_file):
    with open(ivy_file) as infile:
        lines = infile.readlines()

    in_after_init, in_action, in_module, in_invariant = False, False, False, False
    action_str, action_buffer = '', []
    module_str = ''
    invariant_buffer = []
    for line_num, line in enumerate(lines):
        # remove comment suffix
        line = line.split('#')[0].strip()

        if line == '':  # empty line
            pass

        elif in_after_init:  # in after init scope
            if line == '}':
                init_block[:0] = build_initialization_block()
                in_after_init = False
            else:
                init_block.extend(parse_init_stmt(line))

        elif in_action:    # in action scope
            if line == '}':
                parse_action(action_str, action_buffer)
                action_str = ''
                action_buffer.clear()
                in_action = False
            else:
                action_buffer.append(line)

        elif in_module:
            if line == '}':
                in_module = False

        elif in_invariant:
            if line.startswith('invariant'):
                parse_invariant(invariant_buffer)
                invariant_buffer.clear()
                invariant_buffer.append(line[len('invariant') + 1:].strip())
            else:
                invariant_buffer.append(line)


        else:              # in global scope
            if line_num == 0:
                if not line.split(' ') == ['#lang', 'ivy1.7']:
                    print('Ivy file must begin with #lang ivy1.7')
                    exit(-1)

            elif line.startswith('type'):  # type declaration
                type_name = line[len('type') + 1:].strip()
                parse_type(type_name)

            elif line.startswith('relation'):  # relation declaration
                relation_str = line[len('relation')+1:].strip()
                parse_relation(relation_str)

            elif line.startswith('function'):
                function_str = line[len('function') + 1:].strip()
                parse_function(function_str)

            elif line.startswith('individual'):
                individual_str = line[len('individual') + 1:].strip()
                parse_individual(individual_str)

            elif line.startswith('axiom'):
                axiom_str = line[len('axiom') + 1:].strip()
                parse_axiom(axiom_str)

            elif line.startswith('after'):
                words = line.split()
                if words != ['after', 'init', '{']:
                    print('Line {} invalid. Maybe it should be "after init {{" ?'.format(line))
                    exit(-1)
                calc_minimum_sizes()
                calc_type_abbrs()
                resolve_total_orders()
                merge_axioms()
                in_after_init = True

            elif line.startswith('action'):
                action_str = line[len('action')+1:].strip()
                in_action = True

            elif line.startswith('module'):
                in_module = True
                module_str = line[len('module')+1:].split('=')[0].strip()
                if not find_module(module_str):
                    print('Unsupported module {}'.format(module_str))
                    exit(-1)

            elif line.startswith('instantiate'):
                instantiate_str = line[len('instantiate')+1:].strip()
                parse_instantiation(instantiate_str)

            elif line.startswith('export'):
                pass

            elif line.startswith('invariant'):
                in_invariant = True
                invariant_buffer.append(line[len('invariant')+1:].strip())

            else:
                print('Unparsable line: {}'.format(line))
                exit(-1)

    if len(invariant_buffer) > 0:
        parse_invariant(invariant_buffer)


def add_blank_line():
    python_codes.append('')


def get_func_name_line():
    action_name_to_funcs = []
    for action_name in actions:
        action_name_to_funcs.append("'{}': {}".format(action_name, action_name))
        action_name_to_funcs.append("'{}_prec': {}_prec".format(action_name, action_name))
    dict_content = ', '.join(action_name_to_funcs)
    return 'func_from_name = {{{}}}'.format(dict_content)


def calc_minimum_sizes():
    # the user is expected to specify a minimum instance (i.e., initial template) to begin with
    # if not, DistInv estimates it based on relation arity and heuristics
    for relation, type_names in relations.items():
        occurrences = Counter(type_names)
        for type_name, count in occurrences.items():
            types[type_name] = max(types[type_name], count)
    # some types have heuristic rules, you can modify or add your own rules
    if 'node' in types and 'node' not in user_specified_min_size:
        if 'quorum' in types:
            # at least 3 nodes needed to distinguish majority from minority
            types['node'] = max(types['node'], 3)
        else:
            # at least 2 nodes for a network
            types['node'] = max(types['node'], 2)
    if 'key' in types and 'value' in types:
        types['value'] = max(types['value'], 2)
    # some modules have predefined rules
    smodule = 'ring_topology'
    if smodule in instantiations:
        # at least 3 nodes for a ring
        for stype, idv_name in instantiations[smodule]:
            types[stype] = max(types[stype], 3)
    # when the initial size fails (refinement fails), increase template size
    while num_attempt[0] > 0:
        types_copy = types.copy()
        for type_name in sorted(types_copy.keys()):
            minimum_size = types[type_name]
            if num_attempt[0] == 0:
                finished = True
                break
            num_attempt[0] -= 1
            types[type_name] = minimum_size + 1
    # one-to-one mapped types must have same size
    if 'one-to-one-f' in axioms:
        func_name, in_type, out_type = get_one_to_one()
        types[in_type] = max(types[in_type], types[out_type])
        types[out_type] = max(types[in_type], types[out_type])


def calc_type_abbrs():
    types_names = list(types.keys())
    first_char_dict = defaultdict(list)
    for type_name in types_names:
        first_char_dict[type_name[0]].append(type_name)
    for first_char, type_list in first_char_dict.items():
        if len(type_list) == 1:
            type_abbrs[type_list[0]] = first_char.upper()
        else:
            second_char_dict = defaultdict(list)
            for type_name in type_list:
                if len(type_name) == 1:
                    print('Please rename type {}. Initials conflict with other types.'.format(type_name))
                    exit(-1)
                second_char_dict[type_name[1]].append(type_name)
            for second_char, second_type_list in second_char_dict.items():
                if len(second_type_list) > 2:
                    print('Types {} have the same initial. Please rename them.'.format(', '.join(second_type_list)))
                    exit(-1)
                type_abbrs[second_type_list[0]] = first_char.upper() + second_char.upper()


def resolve_total_orders():
    le_relations = []
    for relation_name, relation_signature in relations.items():
        if relation_name == 'le':
            assert('total_order' in instantiations and relation_name in instantiations['total_order'])
            assert(len(relation_signature) == 2 and relation_signature[0] == relation_signature[1])
            type_name = relation_signature[0]
            total_ordered_types.append(type_name)
            le_relations.append(relation_name)
    for relation_name in le_relations:
        del(relations[relation_name])


def merge_axioms():
    # some axioms may collectively express a property. If detected, remove old axioms and set new ones
    relation_pairs_to_remove = []
    if 'cond-closed' in axioms and 'cond-total' in axioms:
        for relation1, relation2 in axioms['cond-closed']:
            if (relation1, relation2) in axioms['cond-total']:
                if 'transitivity' in axioms and relation2 in axioms['transitivity']:
                    if 'reflexivity' in axioms and (relation2, True) in axioms['reflexivity']:
                        if 'symmetry' in axioms and (relation2, False) in axioms['symmetry']:
                            # conditional total order found
                            if 'cond-total-order' not in axioms:
                                axioms['cond-total-order'] = []
                            axioms['cond-total-order'].append((relation1, relation2))
                            relation_pairs_to_remove.append((relation1, relation2))
    for relation1, relation2 in relation_pairs_to_remove:
        axioms['cond-closed'].remove((relation1, relation2))
        axioms['cond-total'].remove((relation1, relation2))
        axioms['transitivity'].remove(relation2)
        axioms['symmetry'].remove((relation2, False))
        axioms['reflexivity'].remove((relation2, True))


def enumerate_predicates():
    for type_name, minimum_size in types.items():
        vars_each_type[type_name] = []
        var_base = type_abbrs[type_name]
        for i in range(minimum_size):
            var_name = var_base + str(i+1)
            vars_each_type[type_name].append(var_name)
    for relation_name, relation_signature in relations.items():
        if 'cond-total-order' in axioms and relation_name in list(chain(*axioms['cond-total-order'])):
            continue
        if len(safety_relations) > 0 and relation_name not in safety_relations:
            continue
        vars_list_list = []
        for type_name in relation_signature:
            vars_list_list.append(vars_each_type[type_name])
        params_list = product(*vars_list_list)
        for params in params_list:
            predicate_columns.append('{}[{}]'.format(relation_name, ','.join(params)))
    one_to_one_f, _, _ = get_one_to_one()
    for function_name, (in_type_list, out_type) in functions.items():
        if function_name != one_to_one_f:  # bijection needs no predicates in trace file
            vars_for_product = []
            for type_name in in_type_list:
                vars_for_product.append(vars_each_type[type_name])
            vars_tuples = list(product(*vars_for_product))
            vars_of_out_type = vars_each_type[out_type]
            for i, in_vars in enumerate(vars_tuples):
                func_term = '{}[{}]'.format(function_name, ','.join(in_vars))
                for out_var in vars_of_out_type:
                    if out_type in total_ordered_types:
                        predicate_columns.append('le({},{})'.format(out_var, func_term))
                    else:
                        predicate_columns.append('{}=={}'.format(func_term, out_var))
                for j, in_vars2 in enumerate(vars_tuples):
                    if i < j:
                        func_term2 = '{}[{}]'.format(function_name, ','.join(in_vars2))
                        if out_type in total_ordered_types:
                            predicate_columns.append('le({},{})'.format(func_term, func_term2))
                            predicate_columns.append('le({},{})'.format(func_term2, func_term))
                        else:
                            predicate_columns.append('{}=={}'.format(func_term, func_term2))
    for relation_name, (idv_name, type_list) in module_relations.items():
        if idv_name == 'ring':
            assert(relation_name == 'btw')
            assert(len(type_list) == 3 and type_list[0] == type_list[1] == type_list[2] and type_list[0] in types)
            element_type = type_list[0]
            assert(len(vars_each_type[element_type]) == 3)
            ele1, ele2, ele3 = vars_each_type[element_type]
            predicate_columns.append('btw[{},{},{}]'.format(ele1, ele2, ele3))
    for idv_name, idv_type in individuals.items():
        if idv_type not in total_ordered_types and idv_type != 'bool':
            for vars_name in vars_each_type[idv_type]:
                predicate_columns.append('{}=={}'.format(idv_name, vars_name))
        elif idv_type == 'bool':
            predicate_columns.append('{}[0]'.format(idv_name))


def build_instance_generator():
    lines = []
    lines.append('def instance_generator():')
    indent_prefix = '\t'
    _, one_to_one_in_type, one_to_one_out_type = get_one_to_one()
    final_lines = []
    for type_name in types:
        if type_name == one_to_one_out_type:
            final_lines.append('{}{}_num = {}_num'.format(indent_prefix, one_to_one_out_type, one_to_one_in_type))
        else:
            # we find that the simple uniform distribution within a given range works very well in practice
            minsize = types[type_name]
            gen_size_stmt = '{}{}_num = rng.integers({}, {})'.format(indent_prefix, type_name, minsize, minsize + 4)
            lines.append(gen_size_stmt)
    lines.extend(final_lines)
    return_stmt = '{}return '.format(indent_prefix) + ', '.join(['{}_num'.format(type_name) for type_name in types])
    lines.append(return_stmt)
    return lines


def get_one_to_one():
    func_name, in_type, out_type = '', '', ''
    if 'one-to-one-f' in axioms:
        func_name = axioms['one-to-one-f']
        in_type, out_type = functions[func_name][0][0], functions[func_name][1]
    return func_name, in_type, out_type


def build_sample_function():
    enumerate_predicates()
    lines = []
    max_iter_const = DEFAULT_MAX_ITER if bottleneck[0] == 0 else (DEFAULT_MAX_ITER*3//4 if bottleneck[0] == 1 else DEFAULT_MAX_ITER//2)
    max_simulation_round = MAX_SIMULATE_ROUND if bottleneck[0] == 0 else (MAX_SIMULATE_ROUND * 3 // 4 if bottleneck[0] == 1 else MAX_SIMULATE_ROUND // 2)
    max_no_progress_simulation_round = MAX_SIMULATE_ROUND_WIHOUT_PROGRESS if bottleneck[0] == 0 else (MAX_SIMULATE_ROUND_WIHOUT_PROGRESS * 3 // 4 if bottleneck[0] == 1 else MAX_SIMULATE_ROUND_WIHOUT_PROGRESS // 2)
    lines.append('def sample(max_iter={}):'.format(max_iter_const))
    indent_prefix = '\t'
    type_numbers = ['{}_num'.format(type_name) for type_name in types]
    relation_names = list(relations.keys())
    individual_names = list(individuals.keys())
    function_names = list(functions.keys())
    partial_function_names = ['{}_f'.format(s) for s in axioms['partial-func'].keys()] if 'partial-func' in axioms else []
    module_relation_names = list(module_relations.keys())
    lines.append('{}global {}'.format(indent_prefix, ', '.join(type_numbers + relation_names + individual_names + function_names + partial_function_names + module_relation_names)))
    lines.append('{}df_data = set()'.format(indent_prefix))
    lines.append('{}stopping_criteria = False'.format(indent_prefix))
    lines.append('{}simulation_round = 0'.format(indent_prefix))
    lines.append('{}df_size_history = [0]'.format(indent_prefix))
    if EXACT_NUM_OF_SAMPLE[0] > 0:
        lines.append('{}exact_sample_number = 0'.format(indent_prefix))
    lines.append('{}while stopping_criteria is False:'.format(indent_prefix))
    indent_prefix += '\t'
    lines.append('{}# protocol initialization'.format(indent_prefix))
    lines.append('{}{} = instance_generator()'.format(indent_prefix, ', '.join(type_numbers)))
    lines.extend([(indent_prefix + init_stmt) for init_stmt in init_block])
    lines.append('')
    action_name_strs = ["'{}'".format(s) for s in actions.keys()]
    lines.append('{}action_pool = [{}]'.format(indent_prefix, ', '.join(action_name_strs)))
    lines.append('{}argument_pool = dict()'.format(indent_prefix))
    for action_name, params in actions.items():
        lines.append("{}argument_pool['{}'] = []".format(indent_prefix, action_name))
        tmp_indent = ''
        param_names = []
        for param_name, param_type in params:
            lines.append('{}{}for {} in range({}_num):'.format(indent_prefix, tmp_indent, param_name, param_type))
            param_names.append(param_name)
            tmp_indent += '\t'
        params_tuple = param_names[0] + ',' if len(param_names) == 1 else ', '.join(param_names)
        lines.append("{}{}argument_pool['{}'].append(({}))".format(indent_prefix, tmp_indent, action_name,
                                                                   params_tuple))
    lines.append('')
    lines.append('{}for curr_iter in range(max_iter):'.format(indent_prefix))
    indent_prefix += '\t'
    select_and_execute_block = get_select_and_execute_python_block()
    lines.extend([(indent_prefix + s) for s in select_and_execute_block])
    if EXACT_NUM_OF_SAMPLE[0] > 0:
        lines.append('{}exact_sample_number += 1'.format(indent_prefix))
    lines.append('')
    lines.append('{}# generate subsamples from the current state (sample)'.format(indent_prefix))
    lines.append('{}for k in range({}):'.format(indent_prefix, SUBSAMPLE_PER_SAMPLE))
    indent_prefix += '\t'
    one_to_one_f, one_to_one_in_type, one_to_one_out_type = get_one_to_one()
    for type_name, minimum_size in types.items():
        if type_name != one_to_one_out_type:
            lines.append('{}{}_indices = rng.choice(list(range({}_num)), {}, replace=False)'.
                         format(indent_prefix, type_name, type_name, minimum_size))
            lines.append('{}{}_indices = sorted({}_indices)'.format(indent_prefix, type_name, type_name))
    for type_name in types:
        lvalue = ', '.join(vars_each_type[type_name]) + ','
        if type_name == one_to_one_out_type:
            continue
        elif type_name in total_ordered_types:
            lines.append('{}{} = {}_indices'.format(indent_prefix, lvalue, type_name))
        elif type_name == one_to_one_in_type and one_to_one_out_type in total_ordered_types:
            # e.g., node -> id
            lines.append('{}{} = {}_indices'.format(indent_prefix, lvalue, type_name))
            lvalue = ', '.join(vars_each_type[one_to_one_out_type]) + ','
            rvalue = ', '.join(['{}[{}]'.format(one_to_one_f, in_var_name) for in_var_name in vars_each_type[one_to_one_in_type]]) + ','
            lines.append('{}{} = {}'.format(indent_prefix, lvalue, rvalue))
            one_to_one_pairs = list(zip(vars_each_type[one_to_one_in_type], vars_each_type[one_to_one_out_type]))
            one_to_one_pairs_str = ', '.join(['({}, {})'.format(var1, var2) for var1, var2 in one_to_one_pairs])
            lines.append('{}tmp_list = [{}]'.format(indent_prefix, one_to_one_pairs_str))
            lines.append('{}tmp_list.sort(key=lambda x: x[1])'.format(indent_prefix))
            lines.append('{}{} = tmp_list'.format(indent_prefix, one_to_one_pairs_str))
        else:
            lines.append('{}for {} in permutations({}_indices):'.format(indent_prefix, lvalue, type_name))
            indent_prefix += '\t'
            if type_name == one_to_one_in_type:
                lvalue = ', '.join(vars_each_type[one_to_one_out_type]) + ','
                rvalue = ', '.join(['{}[{}]'.format(one_to_one_f, in_var_name) for in_var_name in vars_each_type[one_to_one_in_type]]) + ','
                lines.append('{}{} = {}'.format(indent_prefix, lvalue, rvalue))
    predicates_str = ', '.join(predicate_columns)
    predicates_str = translate_remove_le(predicates_str)
    lines.append('{}df_data.add(({}))'.format(indent_prefix, predicates_str))
    indent_prefix = '\t\t'
    lines.append('{}simulation_round += 1'.format(indent_prefix))
    lines.append('{}df_size_history.append(len(df_data))'.format(indent_prefix))
    if EXACT_NUM_OF_SAMPLE[0] > 0:
        lines.append('{}stopping_criteria = exact_sample_number > {}'.format(indent_prefix, EXACT_NUM_OF_SAMPLE[0]))
    else:
        lines.append('{}stopping_criteria = simulation_round > {} or (simulation_round > {} and df_size_history[-1] == df_size_history[-{}])'.format(indent_prefix, max_simulation_round, max_no_progress_simulation_round, max_no_progress_simulation_round + 1))
    lines.append('\treturn list(df_data)')
    return lines


def build_main_function():
    lines = []
    lines.append("if __name__ == '__main__':")
    indent_prefix = '\t'
    lines.append('{}start_time = time.time()'.format(indent_prefix))
    lines.append('{}df_data = sample()'.format(indent_prefix))
    predicates_list_str = ', '.join(["'{}'".format(s) for s in predicate_columns])
    predicates_list_str = predicates_list_str.replace('[0]', '')
    predicates_list_str = predicates_list_str.replace('[', '(').replace(']', ')')
    for relation_name, (idv_name, type_list) in module_relations.items():
        predicates_list_str = predicates_list_str.replace(relation_name, '{}.{}'.format(idv_name, relation_name))
    predicates_list_str = predicates_list_str.replace('==', '=')
    lines.append('{}df = pd.DataFrame(df_data, columns=[{}])'.format(indent_prefix, predicates_list_str))
    lines.append('{}df = df.drop_duplicates().astype(int)'.format(indent_prefix))
    lines.append('{}end_time = time.time()'.format(indent_prefix))
    lines.append("{}df.to_csv('../traces/{}.csv', index=False)".format(indent_prefix, PROBLEM))
    lines.append("{}print('Simulation finished. Trace written to traces/{}.csv')".format(indent_prefix, PROBLEM))
    return lines


def write_python_file(python_file):
    python_codes.extend(get_python_header())
    add_blank_line()
    for action_name in actions:
        python_codes.extend(action_precs[action_name])
        add_blank_line()
        python_codes.extend(action_trans[action_name])
        add_blank_line()
    python_codes.append(get_func_name_line())
    add_blank_line()
    python_codes.extend(build_instance_generator())
    add_blank_line()
    python_codes.extend(build_sample_function())
    add_blank_line()
    python_codes.extend(build_main_function())

    with open(python_file, 'w') as outfile:
        for python_line in python_codes:
            outfile.write(python_line + '\n')


def emit_config_file(config_file):
    config_codes = []
    same_type = []
    total_order = []
    one_to_one_f, one_to_one_in_type, one_to_one_out_type = get_one_to_one()
    for type_name, vars_list in vars_each_type.items():
        if len(vars_list) >= 2:
            if type_name in total_ordered_types:
                total_order.append(vars_list)
            else:
                if type_name == one_to_one_in_type or (type_name == one_to_one_out_type and one_to_one_in_type in total_ordered_types):
                    # we do not need both N1,N2,N3 and ID1,ID2,ID3
                    continue
                same_type.append(vars_list)
    assert(len(total_order) <= 1)
    if len(total_order) == 1:
        config_codes.append('total-order: {}'.format(', '.join(total_order[0])))
    if len(same_type) > 0:
        same_type_str_list = []
        for vars_list in same_type:
            same_type_str_list.append(', '.join(vars_list))
        config_codes.append('same-type: {}'.format('; '.join(same_type_str_list)))
    if 'one-to-one-f' in axioms:
        func_name, in_type, out_type = get_one_to_one()
        config_codes.append('one-to-one-f: {}'.format(func_name))
        one_to_one_pairs = list(zip(vars_each_type[in_type], vars_each_type[out_type]))
        one_to_one_pairs = ['{}, {}'.format(var1, var2) for var1, var2 in one_to_one_pairs]
        config_codes.append('one-to-one: {}'.format('; '.join(one_to_one_pairs)))
    config_codes.append('max-literal: {}'.format(MAX_LITIRAL_INIT))
    if bottleneck[0] == 3:
        config_codes.append('hard: true')
    for inv_str in invariants:
        config_codes.append('safety-property: {}'.format(inv_str))

    with open(config_file, 'w') as outfile:
        for config_line in config_codes:
            outfile.write(config_line + '\n')


def translate_ivy_to_python(PROBLEM):
    parse_ivy_file('../protocols/{}/{}.ivy'.format(PROBLEM, PROBLEM))
    write_python_file('../auto_samplers/{}.py'.format(PROBLEM))
    emit_config_file('../configs/{}.txt'.format(PROBLEM))
    print('Instrumenting finished. Simulation script written to auto_samplers/{}.py'.format(PROBLEM))


if __name__ == '__main__':
    PROBLEM = sys.argv[1]
    num_attempt = [0]

    valid_options = ["min_size=", "num_attempt=", "exact_sample="]
    try:
        args = sys.argv
        opts, args = getopt.getopt(args[2:], '', valid_options)
        for opt, arg in opts:
            if opt == '--min_size':
                size_strs = arg.split()
                for size_str in size_strs:
                    assign_splitted = size_str.split('=')
                    if len(assign_splitted) != 2:
                        print('Ill-formed minimum size. Should be in form --min_size="client=2 server=1"')
                        print('See {}'.format(arg))
                        exit(-1)
                    else:
                        type_name, min_size = assign_splitted[0], int(assign_splitted[1])
                        print('type {} has user-specified minimum size {}'.format(type_name, min_size))
                        user_specified_min_size[type_name] = min_size
            elif opt == '--num_attempt':
                num_attempt[0] = int(arg)
            elif opt == '--exact_sample':
                EXACT_NUM_OF_SAMPLE[0] = int(arg)

    except getopt.GetoptError:
        print('Invalid command-line argument')
        sys.exit(-1)


    translate_ivy_to_python(PROBLEM)
