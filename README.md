## Introduction
- We totally collect 1) 18 data augmentation methods from code data, 2) 7 data augmentation methods from NLP, and 3) 4 methods from graph learning and build this project on the top of [ALERT project](https://github.com/soarsmu/attack-pretrain-models-of-code), [Project_CodeNet](https://github.com/IBM/Project_CodeNet), and [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric). Please refer to these projects for more details.

- We modify all 7 data augmentation methodes from NLP to adapt to source code. 
```
|-Data augmentation methods 
    |-—Files structure
    |    |-Back Translation (BT) --> BT.py
    |    |  
    |    |-Synonym Replacement (SR) --> eda4code.py
    |    | 
    |    |-Random Deletion (RD) --> eda4code.py
    |    | 
    |    |-Random Insertion (RI) --> eda4code.py
    |    | 
    |    |-Random Swap (RS) --> RS.py
    |    | 
    |    |-Mixup Augmentation(WordMixp, SenMixup, and Manifold-Mixup) --> Mixup.py
    |-DONE
```



## Requirements
On Ubuntu:
the following software packages are dependencies and will be installed automatically.
```shell
pip install numpy nltk gensim textblob googletrans 
pip install textaugment
```

- Tasks based on TensorFlow

```shell
Python (>=3.6)
TensorFlow (version 2.3.0) 
Keras (version 2.4.3)
CUDA 10.1
cuDNN (>=7.6)
```

- Tasks based on Pytorch
```shell
Python (>=3.6)
Pytorch (version 1.6.0) 
CUDA 10.1
cuDNN (>=7.6)
```

### Graph Data augmentation methods are implemented by PyG (PyTorch Geometric), a library built upon PyTorch to easily write and train Graph Neural Networks (GNNs)
- Please refer to the [PyG project](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.dropout_edge) for more details. 

## Fine-tuning CodeBERT & GraphCodeBERT 

- pip install torch==1.4.0
- pip install transformers==2.5.0
- pip install filelock

### Fine-Tune 
```shell
cd CodeBERT

python run.py \
    --output_dir=./saved_models \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --num_train_epochs 50 \
    --block_size 256 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --num_labels 250 \  # Number Classifications
    --seed 123456  2>&1 | tee train.log
```

```shell
cd GraphCodeBERT

python run.py \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --config_name microsoft/graphcodebert-base \
    --do_train \
    --num_train_epochs 50 \
    --code_length 384 \
    --data_flow_length 384 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --num_labels 250 \  # Number Classifications
    --seed 123456  2>&1 | tee train.log
```

## Dataset
- Java250: https://developer.ibm.com/exchanges/data/all/project-codenet/
- Python800: https://developer.ibm.com/exchanges/data/all/project-codenet/
- Refactory: https://github.com/githubhuyang/refactory
- CodRep: https://github.com/KTH/CodRep-competition
- Google Code Jam (GCJ): https://drive.google.com/uc?id=1t0lmgVHAVpB1GxVqMXpXdU8ArJEQQfqe
- BigCloneBench: https://github.com/soarsmu/attack-pretrain-models-of-code/blob/main/CodeXGLUE/Clone-detection-BigCloneBench/dataset/data.jsonl
