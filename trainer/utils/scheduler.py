import optax


def constant_lr_schedule(lr, init_lr=None, warmup_steps=None, warmup=False):
    """Creates a constant LR Schedule with optional warmup of `warmup_steps`

    Args:
        lr: an end value for the learning rate during the linear schedule
        init_lr: an initial value for the learning rate during the linear schedule
        warmup_steps: number of steps for the linear schedule
        warmup: if we should use a linear warmup schedule

    Returns:
        optax.schedule that maps steps to LR

    """
    if warmup and (warmup_steps is None or init_lr is None):
        raise ValueError("Must provide warmup steps of warmup is set to True")

    schedule = optax.constant_schedule(value=lr)

    if warmup:
        schedule = warmup_schedule(init_lr, lr, warmup_steps, schedule)

    return schedule


def linear_lr_schedule(lr, end_lr, num_train_steps, init_lr=None, warmup_steps=None, warmup=False):
    """Creates a linear LR Schedule with optional warmup of `warmup_steps`

    Args:
        lr: an end value for the learning rate during the linear schedule
        init_lr: an initial value for the learning rate during the linear schedule
        warmup_steps: number of steps for the linear schedule
        warmup: if we should use a linear warmup schedule

    Returns:
        optax.schedule that maps steps to LR

    """
    if warmup and (warmup_steps is None or init_lr is None):
        raise ValueError("Must provide warmup steps of warmup is set to True")

    if warmup:
        decay_steps = num_train_steps - warmup_steps
        schedule = optax.linear_schedule(
            init_value=lr, end_value=end_lr, transition_steps=decay_steps
        )
        schedule = warmup_schedule(init_lr, lr, warmup_steps, schedule)
    else:
        schedule = optax.linear_schedule(
            init_value=lr, end_value=end_lr, transition_steps=num_train_steps
        )

    return schedule


def warmup_schedule(init_lr, end_lr, warmup_steps, schedule):
    warmup_schedule = optax.linear_schedule(
        init_value=init_lr, end_value=end_lr, transition_steps=warmup_steps
    )

    schedule = optax.join_schedules(
        schedules=[warmup_schedule, schedule], boundaries=[warmup_steps]
    )

    return schedule
