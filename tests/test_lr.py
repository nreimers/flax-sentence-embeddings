import unittest

from trainer.utils.scheduler import lr_schedule


class TestLRSchedule(unittest.TestCase):
    def test_raise_bad_args(self):
        with self.assertRaises(ValueError):
            lr_schedule(2e-5, init_lr=None, warmup=True, warmup_steps=500)
        with self.assertRaises(ValueError):
            lr_schedule(2e-5, init_lr=0, warmup=True, warmup_steps=None)

    def test_schedule_no_warmup(self):
        schedule = lr_schedule(lr=2e-5, init_lr=None, warmup_steps=None, warmup=False)
        lrs = [schedule(i) for i in range(10000)]

        self.assertTrue(all(lr == lrs[0] for lr in lrs))

    def test_schedule_with_warmup(self):
        schedule = lr_schedule(lr=2e-5, init_lr=0, warmup_steps=500, warmup=True)
        lrs = [schedule(i) for i in range(1000)]

        self.assertTrue(all(lrs[i] < 2e-5 and lrs[i] >= 0 for i in range(500)))
        self.assertTrue(all(lrs[i] == 2e-5 for i in range(500, 1000)))
