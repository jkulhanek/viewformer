import json
import torch
import os
import fsspec
import logging
from viewformer.models import load_config, AutoModelTH as AutoModel


def load_model(checkpoint):
    model_path, checkpoint = os.path.split(checkpoint)
    with fsspec.open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = load_config(json.load(f))

    model = AutoModel.from_config(config)
    checkpoint_data = torch.load(os.path.join(model_path, checkpoint), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint_data['state_dict'])
    return model


class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1, verbose=False):
        self.optimizer = optimizer

        self.schedule = schedule
        super().__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'schedule')}
        state_dict['schedule'] = str(self.schedule)
        return state_dict

    def load_state_dict(self, state_dict):
        schedule_str = state_dict.pop('schedule')
        self.__dict__.update(state_dict)
        state_dict['schedule'] = schedule_str

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logging.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.")
        return [self.schedule(self.last_epoch) for _ in self.base_lrs]
