import json
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from dataset.convert_to_token_ids import qwen2_preprocess, batch_convert

from dataset.lmdb_utils import Dataset2Lmdb

class MedicalMix(Dataset):

    def __init__(self,
                 data,
                 tokenizer_path,
                 proc_func,
                 max_seq_len=256,
                 num_worker=10,
                 max_dataset_len=-1,
                 ):
        self.data = data
        self.tokenizer_path = tokenizer_path
        self.proc_func = proc_func
        self.max_seq_len = max_seq_len
        self.num_worker = num_worker
        samples = self._load_raw(data)
        if max_dataset_len > 0:
            samples = samples[0:max_dataset_len]
        self.tokenized_samples = self.convert2tokenids(samples=samples)

    @staticmethod
    def load_texts(data_file, expected_size=None):
        texts = []
        for line in tqdm(open(data_file), total=expected_size, desc=f'Loading {data_file}'):
            texts.append(json.loads(line))
        return texts

    def _load_raw(self, data):
        data_paths = data['data_paths']
        samples = []
        for data_path in data_paths:
            print(f'data_path: {data_path}')
            raw_datas = self.load_texts(data_path)  # [0:10]
            # print(raw_datas)
            # exit(0)
            for data in raw_datas:
                messages = []
                messages.append({"role": "user", "content": data['instruction'] + data['input']})
                messages.append({"role": "assistant", "content": data['output']
                                 })
                # print(messages)
                samples.append(dict(conversations=messages))
        print(f'num_raw_samples: {len(samples)}')
        return samples


    def convert2tokenids(self, samples):
        tokenized_samples = batch_convert(samples=samples,
                                          process_func=self.proc_func,
                                          tokenizer_path=self.tokenizer_path,
                                          process_num=self.num_worker)
        print(f'[PRE-DATASET] tokenized data generated, num_samples={len(tokenized_samples)}')
        return tokenized_samples

    def __len__(self):
        return len(self.tokenized_samples)

    def __getitem__(self, idx):
        # TODO: pad and clip
        return self.tokenized_samples[idx]

    def write2lmdb(self,
                   lmdb_path,
                   key_tag='unkown',
                   num_workers=4):
        ds2lmdb_tool = Dataset2Lmdb(dataset=self,
                                    lmdb_path=lmdb_path,
                                    key_tag=key_tag,
                                    num_workers=num_workers)
        ds2lmdb_tool.write2lmdb(write_frequency=1000)



if __name__=='__main__':

    dataset = MedicalMix(data=dict(
        data_paths=[
            '/data/data/llm_datasets/medical/finetune/train_zh_0.json',
            '/data/data/llm_datasets/medical/finetune/train_en_1.json',
        ]),
        tokenizer_path="/data/Qwen/Qwen2.5-0.5B-Instruct",
        proc_func=qwen2_preprocess,
        # max_dataset_len=100,
    )

    sample = dataset.__getitem__(0)
    print(sample)

    dataset.write2lmdb(lmdb_path='/data/data/llm_datasets/cache/medical_mix_finetune_train_qwen2.5_tokenized_20241216.lmdb',
                       key_tag='train_zh_and_en')

