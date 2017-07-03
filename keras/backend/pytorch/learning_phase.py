from .control_flow import switch


_LEARNING_PHASE = None


def learning_phase():
    return _LEARNING_PHASE


def set_learning_phase(value):
    global _LEARNING_PHASE
    assert value in {0, 1}
    _LEARNING_PHASE = value


def in_train_phase(x, alt, training=None):
    if training is None:
        training = learning_phase()
        uses_learning_phase = True
    else:
        uses_learning_phase = False

    if training:
        if callable(x):
            return x()
        else:
            return x
    else:
        if callable(alt):
            return al()
        else:
            return alt

    x = switch(training, x, alt)
    if uses_learning_phase:
        x._uses_learning_phase = True
    return x


def in_test_phase(x, alt, training=None):
    return in_train_phase(alt, x, training=training)
