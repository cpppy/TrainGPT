# TrainGPT
Instruct Finetune on Qwen2.5-0.5B with DeepSpeed

With max_seq_len=128 and qwen2.5-0.5B-instruct as base model, you can run the code on a single RTX3060 GPU device.  
While, single node with multi GPUs or multi nodes with multi GPUs is also supported by this code.   


## TODO List
1. More Dataset and More Base Models(especially for small LLM)  
2. Pretrain
3. DPO  
4. PPO  


## Training Steps
#### step1. Prepare Dataset(lmdb cache for boosting)
```python
cd dataset & vim generate_lmdb.py   
```
the dataset: alpaca was downloaded from huggingface: yahma/alpaca-cleaned    
if you want to use your own datasets, just modify the script: dataset/alpaca_cleaned.py    

```python
    dataset = AlpacaDataset(data=dict(
        data_paths=[
            '../data/alpaca_data_cleaned.json',  # path to raw data
        ]),
        tokenizer_path="/data/Qwen/Qwen2.5-0.5B-Instruct",
        proc_func=qwen2_preprocess,
        # max_dataset_len=100,
    )

    sample = dataset.__getitem__(0)
    print(sample)

    dataset.write2lmdb(lmdb_path='/data/data/llm_datasets/cache/alpaca_cleaned_instruct_qwen2.5_tokenized_20241221.lmdb',  # cache path for tokenized data
                       key_tag='alpaca_cleaned')
```

#### step2. Finetune Model with Instruct Dataset
setting the arguments below, and then, set up the training code.
```python
train_lmdb_path
eval_lmdb_path
tokenizer_name_or_path
model_name_or_path
```

```python
cd finetune
deepspeed finetune_qwen.py  
```

#### Finetune on a single node with 2 * RTX4090      
<img src="./assets/finetune_on_2x4090_screenshot.png" width="400">  

#### Finetune on 2 nodes(2 * RTX4090 and 2 * A100), extremely slow because a 72B-awq deployed on each A100 device.   
<img src="./assets/finetune_on_2_nodes_screenshot.png" width="400">  

#### Visualize Metrics on Tensorboard
```python
deepspeed finetune_qwen_tb.py
```

<img src="./assets/finetune_tensorboard.png" width="600"> 

## Custom Develop

#### Package: dschat   
the package dschat was copied from microsoft/DeepSpeedExamples with nothing modified.   
the position is "DeepSpeedExamplesapplications/DeepSpeed-Chat/dschat"





## Reference   
https://github.com/microsoft/DeepSpeedExamples




