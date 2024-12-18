import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
import six
import string
import time
import numpy as np

import lmdb
import pickle

import torch
from torch.utils.data.dataset import Dataset

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class LMDBSearch(object):
    def __init__(self, db_path, max_readers=8):
        super(LMDBSearch, self).__init__()
        self.db_path = db_path
        self.max_readers = max_readers

        # prepare initialization
        env = lmdb.open(db_path, subdir=osp.isdir(db_path), max_readers=max_readers,
                        readonly=True, lock=False,
                        readahead=False, meminit=False)
        # print(env.info())
        with env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            # self.length = pa.deserialize(txn.get(b'__len__'))
            # self.keys = pa.deserialize(txn.get(b'__keys__'))
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        print(f'lmdb_len: {self.length}')
        print(f'n_key: {len(self.keys)}')
        print('[DATASET] load data from lmdb file, n_sample: {}'.format(self.length))
        env.close()

    def _open_lmdb(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path), max_readers=self.max_readers,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def search_by_index(self, key):
        # key = index.encode()
        assert key in self.keys

        if not hasattr(self, 'txn'):
            self._open_lmdb()

        byteflow = self.txn.get(key)
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked
        return imgbuf

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class MultiLMDBSearch(object):
    def __init__(self, db_path, max_readers=8):
        super(MultiLMDBSearch, self).__init__()
        if isinstance(db_path, str):
            db_path = [db_path]
        self.db_path = db_path
        self.max_readers = max_readers
        self.ks = []
        self.vs = []

        self.length = 0
        self.keys = []
        self.key2envid = {}
        self.txns = [None] * len(db_path)

        for idx, path in enumerate(db_path):
            # prepare initialization
            env = lmdb.open(path, subdir=osp.isdir(path), max_readers=max_readers,
                            readonly=True, lock=False,
                            readahead=False, meminit=False)
            # print(env.info())
            with env.begin(write=False) as txn:
                # self.length = txn.stat()['entries'] - 1
                # self.length = pa.deserialize(txn.get(b'__len__'))
                # self.keys = pa.deserialize(txn.get(b'__keys__'))
                self.length += pickle.loads(txn.get(b'__len__'))
                _keys = pickle.loads(txn.get(b'__keys__'))
                self.keys.extend(_keys)
                self.key2envid.update({k: idx for k in _keys})
            env.close()

        self._open_lmdb()

        print(f'lmdb_len: {self.length}')
        print(f'n_key: {len(self.keys)}')
        print('[DATASET] finish loading data from all lmdb files, n_sample: {}'.format(self.length))

    def _open_lmdb(self):
        for idx, path in enumerate(self.db_path):
            env = lmdb.open(path, subdir=os.path.isdir(path), max_readers=self.max_readers,
                            readonly=True, lock=False,
                            readahead=False, meminit=False)
            txn = env.begin(write=False)
            self.txns[idx] = txn

    def search_by_index(self, key):
        # key = index.encode()
        assert key in self.keys

        # if not hasattr(self, 'txns'):
        #     self._open_lmdb()

        byteflow = self.txns[self.key2envid[key]].get(key)
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked
        return imgbuf

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + '\n'.join(self.db_path) + ')'


class InstructDataset(Dataset):

    def __init__(self,
                 db_path,
                 max_seq_len=256,
                 pad_token_id=151643,  # qwen2.5 pad_token_id
                 padding_side='right'
                 ):
        super(InstructDataset, self).__init__()
        assert db_path is not None
        # self.lmdb_search = LMDBSearch(db_path=db_path)
        self.lmdb_search = MultiLMDBSearch(db_path=db_path)
        self.keys = self.lmdb_search.keys
        self.len = self.lmdb_search.length
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        assert padding_side in ['left', 'right']
        self.padding_side = padding_side

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.lmdb_search.search_by_index(self.keys[idx])
        # print(sample)
        # exit(0)
        input_ids = sample['input_ids']
        labels = sample['labels']
        max_len = self.max_seq_len
        ################## padding or clip ##################
        if self.padding_side == 'right':
            input_ids += [self.pad_token_id] * max(0, max_len - len(input_ids))
            labels += [IGNORE_TOKEN_ID] * max(0, max_len - len(labels))
        else:
            input_ids = [self.pad_token_id] * max(0, max_len - len(input_ids)) + input_ids
            labels = [IGNORE_TOKEN_ID] * max(0, max_len - len(labels)) + labels
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

        input_ids = torch.tensor(input_ids, dtype=torch.int)
        labels = torch.tensor(labels, dtype=torch.int)
        attention_mask = input_ids.ne(self.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels.type(torch.int64),
            attention_mask=attention_mask,
        )


if __name__ == '__main__':
    dataset = InstructDataset(
        db_path=[
            '/mnt2/data/llm_datasets/cache/medical_mix_part1_train_tokenizerQwen25_cache_20241129.lmdb',
            '/mnt2/data/llm_datasets/cache/medical_mix_part1_eval_tokenizerQwen25_cache_20241129.lmdb',
        ],
        max_seq_len=256,
    )

    print(f'dataset_len: {len(dataset)}')
    print(dataset.__getitem__(0))
