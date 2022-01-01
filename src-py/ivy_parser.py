# an Ivy expression parser, takes a string and outputs a tree


import re


class TreeNode:
    def __init__(self, parent_node):
        self.node_type = None
        self.substr = ''
        self.metadata = None
        self.parent = parent_node
        self.children = []


def strip_parenthesis(in_str):
    # count leading '('
    curr_str = in_str.strip()
    while True:
        if curr_str[0] == '(':
            if curr_str[-1] != ')':
                return curr_str
            count = 1
            for i in range(1, len(curr_str)):
                if curr_str[i] == '(':
                    count += 1
                elif curr_str[i] == ')':
                    count -= 1
                if count == 0:
                    if i == len(curr_str) - 1:
                        curr_str = curr_str[1: -1].strip()
                    else:
                        return curr_str
        else:
            return curr_str


def find_closing_parenthesis(target_str, start_idx):
    assert(target_str[start_idx] == '(')
    count = 1
    for idx in range(start_idx+1, len(target_str)):
        if target_str[idx] == '(':
            count += 1
        elif target_str[idx] == ')':
            count -= 1
        if count == 0:
            return idx
    print('Parenthesis mismatch on string {}'.format(target_str))
    assert(False)


def split_string_with_parenthesis_by_delimeter(in_str, delimeter):
    # if in_str = 'p$(q$r)', delieteter = '$', then output_parts = ['p', '(q$r)']
    parts = in_str.split(delimeter)
    lp_net_total = 0
    buffered_parts = []
    output_parts = []
    for part in parts:
        buffered_parts.append(part)
        lp_net = part.count('(') - part.count(')')
        lp_net_total += lp_net
        if lp_net_total == 0:
            output_parts.append(delimeter.join(buffered_parts))
            buffered_parts.clear()
    assert lp_net_total == 0
    return output_parts

def add_disambiguating_forallexists_parenthesis(in_str):
    settled = ''
    while len(in_str) > 0:
        match_obj = re.search('(forall)|(exists)', in_str)
        if match_obj is None:
            settled += in_str
            break
        else:
            prefix = in_str[: match_obj.start(0)]
            count = 0
            for i in range(match_obj.end(0), len(in_str)):
                if in_str[i] == '(':
                    count += 1
                elif in_str[i] == ')':
                    count -= 1
                if count == -1 or i == len(in_str) - 1:
                    settled += prefix + '(' + in_str[match_obj.start(0): i+1] + ')'
                    in_str = in_str[i+1:]
                    break
    return settled


def parse_comma_params_and_add_children(this_node, params_str):
    output_parts = split_string_with_parenthesis_by_delimeter(params_str, ',')
    for param_str in output_parts:
        this_node.children.append(tree_parse_ivy_expr(param_str.strip(), this_node))


def tree_parse_ivy_expr(ivy_expr, parent_node):
    ivy_expr = strip_parenthesis(ivy_expr).strip()
    this_node = TreeNode(parent_node)
    this_node.substr = ivy_expr
    if 'if' in ivy_expr and 'else' in ivy_expr:
        this_node.node_type = 'if-else'
        if_splitted = ivy_expr.split('if')
        assert len(if_splitted) == 2  # no nested if-else for now
        if_child = tree_parse_ivy_expr(if_splitted[0].strip(), this_node)
        else_splitted = if_splitted[1].split('else')
        assert len(else_splitted) == 2
        branch_condition_child = tree_parse_ivy_expr(else_splitted[0].strip(), this_node)
        else_child = tree_parse_ivy_expr(else_splitted[1].strip(), this_node)
        this_node.children = [if_child, branch_condition_child, else_child]
        return this_node
    if ivy_expr.startswith('forall') or ivy_expr.startswith('exists'):
        this_node.node_type = ivy_expr[:6]
        dot_splitted = ivy_expr[len('forall')+1:].split('.', 1)
        assert(len(dot_splitted) == 2)
        qvars_str, formula = dot_splitted
        qvars_raw = qvars_str.split(',')
        qvars = {}
        for qvar in qvars_raw:
            colon_splitted = qvar.split(':')
            assert(len(colon_splitted) <= 2)
            # Ivy can optionally specify the type of quantified variables 1) forall X 2) forall X:node
            if len(colon_splitted) == 1:
                qvars[colon_splitted[0].strip()] = None
            else:
                qvars[colon_splitted[0].strip()] = colon_splitted[1].strip()
        # qvars = [qvar.strip() for qvar in qvars]
        this_node.metadata = qvars
        this_node.children = [tree_parse_ivy_expr(formula.strip(), this_node)]
        return this_node
    # now the top-level is bound to be a logical formula
    # we should mask the atomic substrings that we should not break into at this level, either from parenthesis or from forall/exists
    # in p(X) & (q(X) | r(X)), (q(X) | r(X)) should be considered atomic, handled in function split_string_with_parenthesis_by_delimeter
    # in p(X) & forall Y. q(X,Y) & r(Y), forall Y. q(X,Y) & r(Y) should be considered an atomic substring, handled in function add_disambiguating_forallexists_parenthesis
    ivy_expr = add_disambiguating_forallexists_parenthesis(ivy_expr)
    equiv_splitted = split_string_with_parenthesis_by_delimeter(ivy_expr, '<->')
    if len(equiv_splitted) >= 2:
        assert len(equiv_splitted) == 2
        this_node.node_type = 'equiv'
        for segment in equiv_splitted:
            this_node.children.append(tree_parse_ivy_expr(segment.strip(), this_node))
        return this_node
    imply_splitted = split_string_with_parenthesis_by_delimeter(ivy_expr, '->')
    if len(imply_splitted) >= 2:
        assert len(imply_splitted) == 2
        this_node.node_type = 'imply'
        for segment in imply_splitted:
            this_node.children.append(tree_parse_ivy_expr(segment.strip(), this_node))
        return this_node
    or_splitted = split_string_with_parenthesis_by_delimeter(ivy_expr, '|')
    if len(or_splitted) >= 2:
        this_node.node_type = 'or'
        for segment in or_splitted:
            this_node.children.append(tree_parse_ivy_expr(segment.strip(), this_node))
        return this_node
    and_splitted = split_string_with_parenthesis_by_delimeter(ivy_expr, '&')
    if len(and_splitted) >= 2:
        this_node.node_type = 'and'
        for segment in and_splitted:
            this_node.children.append(tree_parse_ivy_expr(segment.strip(), this_node))
        return this_node
    nequal_splitted = split_string_with_parenthesis_by_delimeter(ivy_expr, '~=')
    if len(nequal_splitted) >= 2:
        assert (len(nequal_splitted) == 2)
        this_node.node_type = 'nequal'
        for segment in nequal_splitted:
            this_node.children.append(tree_parse_ivy_expr(segment.strip(), this_node))
        return this_node
    equal_splitted = split_string_with_parenthesis_by_delimeter(ivy_expr, '=')
    if len(equal_splitted) >= 2:
        assert(len(equal_splitted) == 2)
        this_node.node_type = 'equal'
        for segment in equal_splitted:
            this_node.children.append(tree_parse_ivy_expr(segment.strip(), this_node))
        return this_node
    if 'A' <= ivy_expr[0] <= 'Z':
        # quantified variable, e.g., ID1
        assert(re.match('[A-Z][a-zA-Z0-9_]*$', ivy_expr) is not None)
        this_node.node_type = 'qvar'
        return this_node
    if 'a' <= ivy_expr[0] <= 'z':
        if re.match('^[a-z][a-z0-9_]*$', ivy_expr) is not None:
            # individual (constant/variable), e.g., zero
            this_node.node_type = 'const'
            return this_node
        else:
            # starts with a relation
            match = re.match('^[a-z][a-z0-9_]*\(', ivy_expr)
            if match is not None:
                right_parenthesis_idx = find_closing_parenthesis(ivy_expr, match.end() - 1)
                if right_parenthesis_idx == len(ivy_expr) - 1:
                    # predicate, e.g., holds_lock(E1, N1)
                    # note: currently, function application idn(n) is parsed as predicate
                    this_node.node_type = 'predicate'
                    this_node.metadata = match.group(0)[:-1]
                    parse_comma_params_and_add_children(this_node, ivy_expr[match.end(): -1].strip())
                    return this_node
            match = re.search('^([a-z]+)\.([a-z]+)\(', ivy_expr)
            if match is not None:
                right_parenthesis_idx = find_closing_parenthesis(ivy_expr, match.end() - 1)
                if right_parenthesis_idx == len(ivy_expr) - 1:
                    # module predicate, e.g., ring.btw(N1,N2,N3)
                    this_node.node_type = 'module_predicate'
                    this_node.metadata = (match.group(1), match.group(2))
                    parse_comma_params_and_add_children(this_node, ivy_expr[match.end(): -1].strip())
                    return this_node
    if ivy_expr.startswith('~'):
        this_node.node_type = 'not'
        this_node.children = [tree_parse_ivy_expr(ivy_expr[1:], this_node)]
        return this_node
    if ivy_expr == '*':
        this_node.node_type = 'star'
        return this_node
    print('Ivy expr {} cannot be parsed'.format(ivy_expr))
    assert False


def all_nodes_of_tree(root):
    if len(root.children) == 0:
        return [root]
    node_list = [root]
    for child in root.children:
        node_list.extend(all_nodes_of_tree(child))
    return node_list


node_order_list = ['star', 'const', 'qvar', 'nequal', 'equal', 'predicate', 'module_predicate', 'not', 'and', 'or', 'imply', 'equiv', 'forall', 'exists', 'if-else']
node_order = {s:i for (i,s) in enumerate(node_order_list)}


def standardize_tree(root):
    num_children = len(root.children)
    if root.node_type in ['and', 'or']:
        for i in range(0, num_children - 1):
            for j in range(0, num_children - i - 1):
                if node_order[root.children[j].node_type] > node_order[root.children[j + 1].node_type]:
                    root.children[j], root.children[j + 1] = root.children[j + 1], root.children[j]
    for child in root.children:
        standardize_tree(child)
    return root


# for debugging this Ivy expr tree parser
if __name__ == '__main__':
    root = tree_parse_ivy_expr('le(X, src(P)) & hello(N1) -> gg(P)', None)
    print('Completed')
