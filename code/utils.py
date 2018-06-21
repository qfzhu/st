import numpy as np
import random

def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
        for sent in sents]

def feed_dictionary(model, batch, rho, gamma, dropout=1, learning_rate=None):
    feed_dict = {model.dropout: dropout,
                 model.learning_rate: learning_rate,
                 model.rho: rho,
                 model.gamma: gamma,
                 model.batch_len: batch['len'],
                 model.batch_size: batch['size'],
                 model.enc_inputs: batch['enc_inputs'],
                 model.dec_inputs: batch['dec_inputs'],
                 model.targets: batch['targets'],
                 model.weights: batch['weights'],
                 model.labels: batch['labels']}
    return feed_dict

def makeup(_x, n):
    x = []
    for i in range(n):
        x.append(_x[i % len(_x)])
    return x

def reorder(order, _x):
    x = range(len(_x))
    for i, a in zip(order, _x):
        x[i] = a
    return x

# noise model from paper "Unsupervised Machine Translation Using Monolingual Corpora Only"
def noise(x, unk, word_drop=0.0, k=3):
    n = len(x)
    for i in range(n):
        if random.random() < word_drop:
            x[i] = unk

    # slight shuffle such that |sigma[i]-i| <= k
    sigma = (np.arange(n) + (k+1) * np.random.rand(n)).argsort()
    return [x[sigma[i]] for i in range(n)]

def get_batch(x, r, y, word2id, noisy=False, min_len=5):
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']

    rev_x, go_x, go_r, x_eos, r_eos, weights = [], [], [], [], [], []
    max_len = max([len(sent) for sent in x])
    max_len = max(max_len, min_len)
    r_max_len = max([len(response) for response in r])
    r_max_len = max(r_max_len, min_len)
    for sent, response in zip(x, r):
        sent_id = [word2id[w] if w in word2id else unk for w in sent]
        response_id = [word2id[w] if w in word2id else unk for w in response]
        l = len(sent)
        rl = len(response)
        padding = [pad] * (max_len - l)
        rpadding = [pad] * (r_max_len - rl)
        _sent_id = noise(sent_id, unk) if noisy else sent_id
        rev_x.append(padding + _sent_id[::-1])
        go_x.append([go] + sent_id + padding)
        go_r.append([go] + response_id + rpadding)
        x_eos.append(sent_id + [eos] + padding)
        r_eos.append(response_id + [eos] + rpadding)
        # weights.append([1.0] * (l+1) + [0.0] * (max_len-l))
        weights.append([1.0] * (rl+1) + [0.0] * (r_max_len-rl))

    return {'enc_inputs': rev_x,
            # 'dec_inputs': go_x,
            # 'targets':    x_eos,
            'dec_inputs': go_r,
            'targets':    r_eos,
            'weights':    weights,
            'labels':     y,
            'size':       len(x),
            'len':        max_len+1}


def get_batches(x0, x1, r0, r1, word2id, batch_size, noisy=False):
    if len(x0) < len(x1):
        x0 = makeup(x0, len(x1))
        r0 = makeup(r0, len(r1))
    if len(x1) < len(x0):
        x1 = makeup(x1, len(x0))
        r1 = makeup(r1, len(r0))
    n = len(x0)

    order0 = range(n)
    z = sorted(zip(order0, x0, r0), key=lambda i: len(i[1]))
    order0, x0, r0 = zip(*z)

    order1 = range(n)
    z = sorted(zip(order1, x1, r1), key=lambda i: len(i[1]))
    order1, x1, r1 = zip(*z)

    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches.append(get_batch(x0[s:t] + x1[s:t], r0[s:t] + r1[s:t],
            [0]*(t-s) + [1]*(t-s), word2id, noisy))
        s = t

    return batches, order0, order1
