from datasets import list_datasets, load_dataset
from pprint import pprint
import pickle
from utils_func import preprocessing
import os



def download(state='show', split=None, path=None):
    if state == 'show':
        pprint(list_datasets(), compact=True)
    else:
        if split == None:
            dataset = load_dataset(state)
            dataset.cleanup_cache_files()
            print('Please select split...\n', dataset)
        else:
            dataset = load_dataset(state, split=split)
            with open(path, 'wb') as f:
                pickle.dump(dataset, f)
            print('{} of {} data are saved'.format(len(dataset), split))
            dataset.cleanup_cache_files()


    
def preprocessing_data(split):
    raw_path = 'data/IMDb/raw/imdb.' + split
    save_path = 'data/IMDb/processed/imdb.' + split
    with open(raw_path, 'rb') as f:
        data = pickle.load(f)

    # data pre-processing
    processed_data = []
    for d in data:
        d['text'] = preprocessing(d['text'])
        processed_data.append([d['text'], d['label']])

    with open(save_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    os.remove(raw_path)