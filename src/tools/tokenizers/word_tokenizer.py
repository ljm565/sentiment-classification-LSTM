from collections import Counter
from utils.func_utils import preprocessing
from utils.filesys_utils import read_dataset



class WordTokenizer:
    def __init__(self, config, data_path):
        self.vocab_size = config.vocab_size
        self.pad_token, self.bos_token, self.unk_token = '[PAD]', '[BOS]', '[UNK]'
        self.pad_token_id, self.bos_token_id, self.unk_token_id = 0, 1, 2
        self.word2idx = {self.pad_token: self.pad_token_id, self.bos_token: self.bos_token_id, self.unk_token: self.unk_token_id}
        self.idx2word = {self.pad_token_id: self.pad_token, self.bos_token_id: self.bos_token, self.unk_token_id: self.unk_token}

        # count the word frequency
        self.word_freq = Counter()
        for s in [d[0].split() for d in read_dataset(data_path)]:
            self.word_freq.update(s)

        # update vocab
        for word, _ in self.word_freq.most_common(self.vocab_size-len(self.word2idx)):
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

        assert len(self.word2idx) == len(self.idx2word)
        self.vocab_size = min(self.vocab_size, len(self.word2idx))


    def tokenize(self, s):
        s = preprocessing(s)
        return s.split()


    def encode(self, s):
        s = [self.word2idx[self.bos_token]] + [self.word2idx[w] if w in self.word2idx else self.word2idx[self.unk_token] for w in self.tokenize(s)]
        return s


    def decode(self, tok):
        s = [self.idx2word[t] for t in tok]
        try:
            s = ' '.join(s[1:tok.index(self.pad_token_id)])
        except ValueError:
            s = ' '.join(s[1:])
        return s