import unittest

from trainer.utils.scheduler import constant_lr_schedule, linear_lr_schedule


class TestLRSchedule(unittest.TestCase):
    def test_raise_bad_args(self):
        with self.assertRaises(ValueError):
            constant_lr_schedule(2e-5, init_lr=None, warmup=True, warmup_steps=500)
            linear_lr_schedule(2e-5, init_lr=None, warmup=True, warmup_steps=500)
        with self.assertRaises(ValueError):
            constant_lr_schedule(2e-5, init_lr=0, warmup=True, warmup_steps=None)
            linear_lr_schedule(2e-5, init_lr=0, warmup=True, warmup_steps=None)

    def test_constant_schedule_no_warmup(self):
        schedule = constant_lr_schedule(lr=2e-5, init_lr=None, warmup_steps=None, warmup=False)
        lrs = [schedule(i) for i in range(10000)]

        self.assertTrue(all(lr == lrs[0] for lr in lrs))

    def test_constant_schedule_with_warmup(self):
        schedule = constant_lr_schedule(lr=2e-5, init_lr=0, warmup_steps=500, warmup=True)
        lrs = [schedule(i) for i in range(1000)]

        self.assertTrue(all(lrs[i] < 2e-5 and lrs[i] >= 0 for i in range(500)))
        self.assertTrue(all(lrs[i] == 2e-5 for i in range(500, 1000)))

    def test_linear_schedule_no_warmup(self):
        num_steps = 1000

        schedule = linear_lr_schedule(lr=1e-3, end_lr=1e-5, num_train_steps=num_steps)

        lrs = [schedule(i) for i in range(num_steps)]

        self.assertTrue(all(lrs[i] <= 1e-3 and lrs[i] >= 1e-5 for i in range(num_steps)))

    def test_linear_schedule_with_warmup(self):
        num_steps = 1000
        decay_steps = 500

        schedule = linear_lr_schedule(
            lr=1e-3,
            end_lr=1e-5,
            num_train_steps=num_steps,
            warmup_steps=500,
            init_lr=0,
            warmup=True,
        )

        lrs = [schedule(i) for i in range(num_steps)]

        self.assertTrue(all(lrs[i] >= 0 and lrs[i] <= 1e-3 for i in range(decay_steps)))
        self.assertTrue(
            all(lrs[i] <= 1e-3 and lrs[i] >= 1e-5 for i in range(decay_steps, num_steps))
        )
