# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.misc
import torchvision


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def histo_summary(self, tag, value, step):
        self.writer.add_histogram(tag, value, step)

    def close(self):
        self.writer.close()
