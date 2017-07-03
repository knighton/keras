def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):
    raise NotImplementedError


def switch(condition, then_expression, else_expression):
    if callable(then_expression):
        then = then_expression
    else:
        then = lambda: then_expression

    if callable(else_expression):
        else_ = else_expression
    else:
        else_ = lambda: else_expression

    if bool(condition):
        return then
    else:
        return else_
