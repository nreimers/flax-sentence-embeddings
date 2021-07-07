import optax


def lr_schedule(lr, init_lr=None, warmup_steps=None, warmup=False):
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
        warmup_schedule = optax.linear_schedule(
            init_value=init_lr, end_value=lr, transition_steps=warmup_steps
        )

        schedule = optax.join_schedules(
            schedules=[warmup_schedule, schedule], boundaries=[warmup_steps]
        )

    return schedule
