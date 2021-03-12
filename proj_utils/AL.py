import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import util.misc as utils

from torch.utils.data import DataLoader, DistributedSampler, Subset, ConcatDataset

from tqdm import tqdm


class ActiveLearner:
    def __init__(self, model, criterion, optimizer, al_method, model_train, dataset_train, dataset_val, device,
                 batch_size=2, initial_size=2500, query_size=2500, epochs_per_query=1, distributed=False,
                 num_workers=1, model_name="DETR"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.al_method = al_method

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.batch_size = batch_size
        # a functions that trains the model for one epoch
        self.model_train = model_train
        self.device = device

        self.history = {}

        self.initial_size = initial_size
        self.current_size = initial_size
        self.total_ds_size = len(dataset_train)
        self.query_size = query_size
        self.epochs_per_query = epochs_per_query

    def init_ds(self, dataset, dataset_size):
        idx = np.random.choice(range(0, len(dataset)), size=dataset_size)
        return Subset(dataset, idx)

    def make_val_loader(self, dataset_val):
        if self.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if self.model_name == "DETR":
            data_loader_val = DataLoader(dataset_val, self.batch_size, sampler=sampler_val,
                                         drop_last=False, collate_fn=utils.collate_fn, num_workers=self.num_workers)
        return data_loader_val

    def make_train_loader(self, dataset_train):
        if self.distributed:
            sampler_train = DistributedSampler(dataset_train)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, self.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=num_workers)
        return data_loader_train

    def query_oracle(self, idx):
        return Subset(self.dataset_train, idx)


    def train_al(self):
        # train the model with active learning
        # TODO: add things to history

        loader_val = self.make_val_loader(self.dataset_val)

        # train on the current labeled dataset

        # AL loop
        n_iters = len(self.dataset_train) // self.query_size
        for i in range(n_iters):
            # TODO - change for queries per epoch changes > 1, for DETR this will train for one epoch
            # train on the current dataset
            if i == 0:
                dataset_sub = self.init_ds(self.dataset_train, self.initial_size)
            else:
                # query_oracle and and expand the current training set
                idx = self.al_method(self.model, self.dataset_train)
                new_samples = self.query_oracle(idx)
                dataset_sub = ConcatDataset([dataset_sub, new_samples])

            loader_subtrain = self.make_train_loader(dataset_sub)

            self.model_train(self.model, self.criterion, loader_subtrain, self.optimizer, self.device, epoch=i)
            print("Trained on: {}/{}".format(len(dataset_sub, len(self.dataset_train))))

        print("[AL]: done")



















