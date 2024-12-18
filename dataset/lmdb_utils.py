import os
import pickle
import lmdb
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

class Dataset2Lmdb(object):

    def __init__(self, dataset, lmdb_path, key_tag='unkown', num_workers=4):
        super(Dataset2Lmdb, self).__init__()
        self.dataset = dataset
        self.data_loader = DataLoader(dataset=dataset,
                                      batch_size=1,
                                      num_workers=num_workers,
                                      collate_fn=lambda x: x)
        self.lmdb_path = lmdb_path
        self.key_tag = key_tag

    @staticmethod
    def dumps_pyarrow(obj):
        """
        Serialize an object.

        Returns:
            Implementation-dependent bytes-like object
        """
        # return pa.serialize(obj).to_buffer()
        return pickle.dumps(obj)

    def write2lmdb(self, write_frequency=5000):
        print("Generate LMDB to %s" % self.lmdb_path)
        isdir = os.path.isdir(self.lmdb_path)
        db = lmdb.open(self.lmdb_path, subdir=isdir,
                       map_size=1099511627776, readonly=False,
                       meminit=False, map_async=True)

        pbar = tqdm(desc='write2lmdb', total=len(self.data_loader))
        txn = db.begin(write=True)
        for idx, batch in enumerate(self.data_loader):
            sample = batch[0]
            # print(type(data), data)
            input_ids = sample['input_ids']
            labels = sample['labels']
            sample_data = dict(input_ids=input_ids, labels=labels)
            txn.put(u'{}'.format(f'{self.key_tag}_idx_{idx}').encode('ascii'), self.dumps_pyarrow(sample_data))
            if idx % write_frequency == 0:
                print("[%d/%d]" % (idx, len(self.data_loader)))
                txn.commit()
                txn = db.begin(write=True)
            pbar.update(1)
        pbar.close()

        # finish iterating through dataset
        txn.commit()

        keys = [u'{}'.format(f'{self.key_tag}_idx_{i}').encode('ascii') for i in range(len(self.dataset))]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', self.dumps_pyarrow(keys))
            txn.put(b'__len__', self.dumps_pyarrow(len(keys)))

        print("Flushing database ...")
        db.sync()
        db.close()