### Installation
1. Git clone our repository and creating conda environment:
```
conda create -n LLMenv python=3.8
conda activate LLMenv
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --upgrade transformers
conda install psutil
pip install peft
pip install sentencepiece
pip install deepspeed=0.14.0
```

```shell
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
```


Successfully installed deepspeed==0.14.0  
transformers:
Successfully installed safetensors-0.4.3 tokenizers-0.19.1 transformers-4.41.2

### Train
```
bash LoraTrainer/train.sh
```
