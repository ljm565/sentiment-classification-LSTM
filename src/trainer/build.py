import os

import torch
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, distributed, random_split

from models import SentimentLSTM
from tools import IMDbDownloader
from utils import RANK, LOGGER, colorstr
from utils.data_utils import DLoader, CustomDLoader, seed_worker

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_model(config, tokenizer, device):
    model = SentimentLSTM(config, tokenizer, device)
    return model.to(device)


def build_dataset(config, tokenizer, modes):
    if config.IMDb_train:
        downloader = IMDbDownloader(config)
        trainset, testset = downloader()
        tmp_dsets = {'train': trainset, 'validation': testset}
        dataset_dict = {mode: DLoader(config, tmp_dsets[mode], tokenizer) for mode in modes}
    else:
        dataset_dict = {mode: CustomDLoader(config.CUSTOM.get(f'{mode}_data_path')) for mode in modes}
    return dataset_dict


def build_dataloader(dataset, batch, workers, shuffle=True, is_ddp=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if not is_ddp else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def get_data_loader(config, tokenizer, modes, is_ddp=False):
    datasets = build_dataset(config, tokenizer, modes)
    dataloaders = {m: build_dataloader(datasets[m], 
                                       config.batch_size, 
                                       config.workers, 
                                       shuffle=(m == 'train'), 
                                       is_ddp=is_ddp) for m in modes}

    return dataloaders