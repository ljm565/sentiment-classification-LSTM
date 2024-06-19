
import os
from pprint import pprint
from datasets import list_datasets, load_dataset

from utils import LOGGER
from utils.func_utils import preprocessing
from utils.filesys_utils import read_dataset, write_dataset



class IMDbDownloader:
    def __init__(self, config):
        self.data_dir = config.IMDb.path
        self.splits = ['train', 'test']


    def download(self, state='show', split=None, path=None):
        if state == 'show':
            pprint(list_datasets(), compact=True)
        else:
            if split == None:
                dataset = load_dataset(state)
                dataset.cleanup_cache_files()
                LOGGER.info('Please select split...\n', dataset)
            else:
                dataset = load_dataset(state, split=split)
                write_dataset(path, dataset)
                LOGGER.info('{} of {} data are saved'.format(len(dataset), split))
                dataset.cleanup_cache_files()


    def preprocessing_data(self, read_path, write_path):        
        # data pre-processing
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        data = read_dataset(read_path)
        processed_data = [[preprocessing(d['text']), d['label']] for d in data]
        write_dataset(write_path, processed_data)
        return processed_data
        

    @staticmethod
    def is_exist(path):
        return os.path.isfile(path)

    
    def __call__(self):
        raw_trainset_path, raw_testset_path = os.path.join(self.data_dir, 'IMDb/raw/imdb.train'), os.path.join(self.data_dir, 'IMDb/raw/imdb.test')
        pp_trainset_path, pp_testset_path = os.path.join(self.data_dir, 'IMDb/processed/imdb.train'), os.path.join(self.data_dir, 'IMDb/processed/imdb.test')

        if not (self.is_exist(pp_trainset_path) and self.is_exist(pp_testset_path)):
            if not (self.is_exist(raw_trainset_path) and self.is_exist(raw_testset_path)):
                LOGGER.info('Downloading IMDb dataset..')
                os.makedirs(os.path.dirname(raw_trainset_path), exist_ok=True)
                os.makedirs(os.path.dirname(raw_testset_path), exist_ok=True)
                for split, path in zip(self.splits, [raw_trainset_path, raw_testset_path]):
                    self.download('imdb', split, path)

            LOGGER.info('Pre-processing the raw IMDb dataset..')
            trainset = self.preprocessing_data(raw_trainset_path, pp_trainset_path)
            testset = self.preprocessing_data(raw_testset_path, pp_testset_path)

            return trainset, testset
        
        return read_dataset(pp_trainset_path), read_dataset(pp_testset_path)
