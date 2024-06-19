import os
import re
import random
import matplotlib.pyplot as plt



def preprocessing(s):
    s = s.replace('<br /><br />', ' ')
    s = re.sub('[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', '', s).lower()
    return s


def visualize_attn(path, all_x, all_y, all_pred, all_attn, tokenizer, positive_threshold):
    all_x = all_x.detach().cpu().numpy()
    all_y = all_y.detach().cpu().numpy()
    all_pred = all_pred.detach().cpu().numpy()
    all_attn = all_attn.detach().cpu().numpy()
    idx, sentiment = [], {0: 'negative', 1: 'positive'}

    for i, (x, y, p) in enumerate(zip(all_x.tolist(), all_y.tolist(), all_pred.tolist())):
        try:
            if len(tokenizer.decode(x).split()) < 20 and ((y == 1 and p >= positive_threshold) or (y == 0 and p < positive_threshold)):
                idx.append(i)
        except ValueError:
            continue

    idx = random.choice(idx)
    x, y, p, a = all_x[idx].tolist(), all_y[idx].tolist(), all_pred[idx].tolist(), all_attn[idx].tolist()
    sentence = tokenizer.decode(x).split()
    score = a[1:x.index(0)]
    assert len(sentence) == len(score)

    idx = list(range(len(score)))
    plt.figure(figsize=(12, 8))
    plt.title('Attention Score ({} review)'.format(sentiment[int(y)]), fontsize=20)
    plt.bar(idx, score)
    plt.xticks(idx, sentence)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'attention_score.jpg'))