GRAMMAR_V0 = """
    <program> ::= <min_stmt_lst> <stmt_lst>
    <stmt_lst> ::= <stmt> | <stmt> <stmt_lst>
    <min_stmt_lst> ::= <stmt> <stmt> <stmt>

    <stmt> ::= <assignment> | <if_block> | <while_block> | <list_op>

    # assignment
    <assignment> ::= <num_assignment> | <bool_assignment>
    <num_assignment> ::=  <var> "=" <expr>
    <bool_assignment> ::= <bool_var> "=" <bool_expr>

    # if_block
    <if_block> ::= "if" <bool_var> ":" "\n" <stmt_lst>

    # while_block
    <init_cnter> ::= "cnter" "=" "0" "\n"      
    <increment_cnter> ::= "cnter" "=" "cnter" "+" <cnter_increment_number> "\n"
    <while_cond,1> ::= <while_cond_var,1> "=" "cnter" "!=" <while_cond_number> "\n"
    <while_block> ::= <while_block_nh> | <init_cnter> <while_cond,1> "while" <while_cond_var,2> ":" "\n" <stmt_lst> <increment_cnter> <while_cond,2>

    # while_block_nh
    <while_block_nh> ::= "while" "True" ":"  "\n" <stmt_lst>

    # list_op
    <list_op> ::= <list_var> ".append(" <operand> ")" | <list_var> ".pop()"
                     
    # expr
    <operand> ::= <number> | <var>
    <list_index> ::= <list_index_number> | <var>
    <expr> ::= <operand> | <arithm_expr> | <list_var> "[" <list_index> "]"
    <arithm_expr> ::= <operand> "+" <number> | <number> "+" <operand> | <operand> "-" <number> | <number> "-" <operand>
    <comp_op> ::= "==" | "!=" 
    <bool_expr> ::= <operand> <comp_op> <number> | <number> <comp_op> <operand>
"""

GRAMMAR_MAP = {
    "grammar_v0": GRAMMAR_V0,
}