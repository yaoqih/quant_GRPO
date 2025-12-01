git clone https://github.com/yaoqih/quant_GRPO
sudo apt-get install aria2 -y
aria2c -x 16 -s 16 https://img.blenet.top/file/vast-ai-volumn/qlib_1d.zip
# aria2c -x 16 -s 16 https://f005.backblazeb2.com/file/vast-ai-volumn/qlib_1d.zip
unzip qlib_1d.zip -d ./quant_GRPO/data/
# pip install blinker cryptography --ignore-installed -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyqlib wandb awscli 
cd quant_GRPO
python -m pipelines.transformer_grpo.trainer --config pipelines/transformer_grpo/config_cn_t1.yaml
# fee8445a6ea62fb7db5fadd1321f5b0f6b5420a6
