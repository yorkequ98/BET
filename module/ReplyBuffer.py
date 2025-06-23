import random
from module import config
import torch


class ReplyBuffer:

    def __init__(self):
        self.action_buffer = []
        self.metric_buffer = []
        self.buffer_counter = 0
        self.batch_size = min(self.buffer_counter, 32)

    def sample(self):
        rand_index = random.randint(0, self.buffer_counter - 1)
        action = self.action_buffer[rand_index]
        metric = self.metric_buffer[rand_index]
        metric = torch.tensor(metric, dtype=torch.float32).unsqueeze(-1).to(config.device)
        return action, metric

    def add(self, action, metric):
        self.action_buffer.append(action)
        self.metric_buffer.append(metric)
        self.buffer_counter += 1

    def __len__(self):
        return len(self.action_buffer)
