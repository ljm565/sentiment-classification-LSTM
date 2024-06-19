import os
import re
import random
import matplotlib.pyplot as plt

import torch



def preprocessing(s):
    s = s.replace('<br /><br />', ' ')
    s = re.sub('[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', '', s).lower()
    return s


def visualize_attn(path, data4vis, tokenizer, positive_threshold, result_num=10):
    all_x = torch.cat(data4vis['x'], dim=0).tolist()
    all_y = torch.cat(data4vis['y'], dim=0).tolist()
    all_pred = torch.cat(data4vis['pred'], dim=0).tolist()
    all_attn = torch.cat(data4vis['attn'], dim=0).tolist()
    idx, sentiment = [], {0: 'negative', 1: 'positive'}

    for i, (x, y, p) in enumerate(zip(all_x, all_y, all_pred)):
        try:
            if len(tokenizer.decode(x).split()) < 20 and ((y == 1 and p >= positive_threshold) or (y == 0 and p < positive_threshold)):
                idx.append(i)
        except ValueError:
            continue
    
    idx = random.sample(idx, min(len(idx), result_num))
    for i, id in enumerate(idx):
        x, y, p, a = all_x[id], all_y[id], all_pred[id], all_attn[id]
        sentence = tokenizer.decode(x).split()
        score = a[1:x.index(0)]
        assert len(sentence) == len(score)

        l = list(range(len(score)))
        plt.figure(figsize=(12, 8))
        plt.title('Attention Score ({} review)'.format(sentiment[int(y)]), fontsize=20)
        plt.bar(l, score)
        plt.xticks(l, sentence)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'attention_score_{i}.jpg'))