# Wph-magic
存储Wph的个人资源

## 电子书网站
https://zh.persian-books.sk/

## work
```python
export PYTHONPATH=/mnt/public/guozhen/test_robotwin/RLinf:/mnt/public/guozhen/test_robotwin/RLinf_RoboTwin:$PYTHONPATH
source switch_env openvla-oft
```

### tmux使用
```python
###### 创建指定名称的会话
tmux new-session -s 会话名称
###### 切换到指定会话
tmux a -t 会话名称
###### 关闭会话
exit # 会话内输入指令
tmux kill-session -t 会话名称  # 会话外输入指令

# 鼠标滚动
nano ~/.tmux.conf
set -g mouse on
tmux source-file ~/.tmux.conf
```


## Proxy配置
```python
nano ~/.bashrc
# 添加字段
alias proxy="export http_proxy=http://127.0.0.1:7897;export https_proxy=http://127.0.0.1:7897;export all_proxy=socks5://127.0.0.1:7897"
alias unproxy="unset http_proxy;unset https_proxy;unset all_proxy"
proxy

```
## ssh rsync 公钥配置
```python
# 生成本地公钥
ssh-keygen -t ed25519 -C "你的邮箱（可选，用于标识密钥）"
ssh-keygen -t rsa -b 4096 -C "你的邮箱（可选）"

# 查看本地公钥
cat ~/.ssh/id_ed25519.pub
cat ~/.ssh/id_rsa.pub

# 向服务器添加公钥
nano ~/.ssh/authorized_keys
```
## 快速向服务器传输文件，可能需要在同一个子网下
```python
# 使用scp(已启用)，使用rsync(二者用法一致)
## 向服务器传输文件(同一子网)I
scp -r 本机目录 username@IP:远程主机目录
rsync -r 本机目录 username@IP:远程主机目录
示例：
scp -r F:\NowData\数据增强\new wph@172.31.71.192:/media/data/workplace_wph
# 先启动虚拟机
wsl -d Ubuntu
rsync -r F:\NowData\数据增强\new wph@172.31.71.192:/media/data/workplace_wph
## 向本机传输文件(同一子网)I
scp -r username@IP:远程主机目录 本机目录
rsync -r username@IP:远程主机目录 本机目录 
示例：
scp -r wph@172.31.71.192:/media/data/workplace_wph F:\NowData\数据增强\new
# 先启动虚拟机，wsl可能只支持/ 不支持 \
wsl -d Ubuntu
rsync -r -P -e "ssh -p 2222" wph@172.31.226.165:/data1/wph/SimpleVLA-RL/models /mnt/c/Users/Wph/Desktop/models        # /mnt/c/Users/Wph/Desktop/models对应本地的C:/Users/Wph/Desktop/models
rsync -r wph@172.31.71.192:/media/data/workplace_wph F:\NowData\数据增强\new
```

## 豆包阅读英文文献指令
```
请你阅读并理解这篇文献，然后将该篇文章的标题作为一级标题，将摘要和各个大标题作为二级标题，将小标题作为三级标题，将小标题下每一部分内容作为四级标题，给我以markdown的语言输出中文的翻译。去掉致谢和参考文献的部分。
```
```
请你以“该篇文章的笔记"为一级标题，分别以“1.该篇文章的研究目的"、"2.该篇文章的研究方法”、“3.该篇文章的研究内容”、“4.该篇文章的最大创新点”、“5.该篇文章给我们的启发"为二级标题，总结的这五部分内容作为三级表题，给我总结后以markdown的语吉输出。
```

## 为服务器配置代理
```python
git clone --branch master --depth 1 https://gh-proxy.com/https://github.com/nelvko/clash-for-linux-install.git \
  && cd clash-for-linux-install \
  && sudo bash install.sh

# 在输入VPN节点的订阅链接即可启动
```

## Docker使用
```python
# 先查看所有运行中的容器，找到要停止的容器ID或名称
docker ps

# 进入已经运行的镜像 
docker exec -it 容器名称或ID bash

# 停止指定容器（使用容器ID或名称）
docker stop <容器ID或容器名称>

# 示例：停止ID为abc123的容器
docker stop abc123

# 强制停止容器（类似kill命令）
docker kill <容器ID或容器名称>

# 保存新的镜像
docker commit 旧镜像名 新镜像名:版本号

# 删除镜像
docker rmi 镜像名称

# 直接以root进入镜像
docker exec -it -u root EmbodiedGen-wph bash

# 运行isaaclab镜像
docker run --name isaac-lab --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v /data1/wph/robot:/workspace/robot:rw \
    isaac-lab:v1

docker run --name isaac-lab --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v /home/zbz/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v /home/zbz/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v /home/zbz/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v /home/zbz/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v /home/zbz/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v /home/zbz/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v /home/zbz/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v /home/zbz/docker/isaac-sim/documents:/data1/wph/robot:rw \
    isaac-lab:v1

# 根据Dockerfile构建镜像
docker build -t 镜像名:版本号 . (当前目录)
# eg:
docker build -t isaac-lab:v0 .
# Dockefile 文件内容示例
FROM nvcr.io/nvidia/isaac-lab:2.1.0            # 基于某个镜像
RUN /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install jupyter  # 安装jupyter
ENV PATH="/isaac-sim/kit/python/bin:${PATH}"        # 自动设置环境变量

# 查看所有容器
docker container ls -a
# 删除指定名称的容器
docker rm -f 容器名称
```

## Jupyter启动isaaclab
```bash
jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root
```

## 具身智能学习路线（长期工作！！！）
https://yv6uc1awtjc.feishu.cn/wiki/KuXTwBY45iRuvokxiIwcHEnVnsh

## MCP基础介绍与使用
https://zihao-ai.feishu.cn/wiki/WhQlwydkMieX4Tki7lbcHK6TnUe

## LLM快速学习（datawhale）
#### LLM部署使用学习
https://github.com/datawhalechina/self-llm
#### LLM原理学习
https://github.com/datawhalechina/happy-llm
#### LLM应用学习
https://github.com/datawhalechina/llm-universe

### DL may be work
http://www.weixiushen.com/project/CNNTricks/CNNTricks.html

#### Ditital Ocean密码
WPH123456789SZU


### huggingface cli 下载数据集
```python
#### 依赖安装
pip install huggingface_hub[cli]        # way 1: base den install 
pip install "huggingface_hub[cli,fast_download]"    # way 2: support bigfile install
#### 配置镜像网站
###### Linux / Ubuntu
export HF_ENDPOINT=https://hf-mirror.com
###### Windows: CMD / PowerShell
set HF_ENDPOINT=https://hf-mirror.com        # CMD
$env:HF_ENDPOINT = "https://hf-mirror.com"   # PowerShell
#### 数据集下载
huggingface-cli download --repo-type dataset --resume-download 复制数据集名称 --local-dir 下载路径
#### 模型checkpoint下载
huggingface-cli download --resume-download [huggingface对应仓库名字] --local-dir [默认使用仓库名字作保存]
# eg: huggingface-cli download --resume-download stabilityai/stable-diffusion-3.5-medium --local-dir stabilityai/stable-diffusion-3.5-medium

### new 指定下载某个文件
hf download \
  --repo-type dataset \
  TianxingChen/RoboTwin2.0 \
  dataset/place_empty_cup/aloha-agilex_clean_50.zip \
  --local-dir ./datasets/place_empty_cup
```

### 单机多卡训练，关闭所有进程
```python
nvidia-smi -i 0 | grep 'python' | awk '{print $5}' | xargs -n1 kill -9
```

### 存储路径
/mnt/nvme/yfyuan/LZL/jacobian/wph


### 运行RDT测试
```python
cd RDT Repo
python -m eval_sim.eval_maniskill --pretrained_path /home/wph/RoboticsDiffusionTransformer/eval_sim/rdt/mp_rank_00_model_states.pt
```

### IsaacLab启动
```python
#### 流传输模式启动
cd /home/lzl/Projects/Isaac/envs      # 进入isaac文件夹
source ./isaaclab/bin/activate        # 激活 uv 环境
# 或者直接
source /home/lzl/Projects/Isaac/envs/isaaclab/bin/activate
# 启动UI
PUBLIC_IP=172.31.226.165
sudo /data1/lzl/Projects/Isaac/envs/isaaclab/bin/isaacsim \
  isaacsim.exp.full.streaming --no-window \
  --/app/livestream/publicEndpointAddress="${PUBLIC_IP}" \
  --/app/livestream/port=49100 --allow-root

#### 直接使用python方式启动
sudo /home/lzl/Projects/Isaac/envs/isaaclab/bin/python3 -m pip list     # 第一种：使用完整的python路径
# 第二种：设置临时环境变量
ISAAC_PYTHON="/home/lzl/Projects/Isaac/envs/isaaclab/bin/python3"
sudo $ISAAC_PYTHON -m pip list

#### 启动脚本的UI可视化
sudo $ISAAC_PYTHON scripts/rsl_rl/random_agent.py  --task=Template-Isaac-Manipulation-Franka-Play-v0  \
--livestream 2  --kit_args="--/app/livestream/publicEndpointAddress=${PUBLIC_IP} --/app/livestream/port=49100"

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/rsl_rl/test.py \                                                                             
  --task=Template-Isaac-Manipulation-Franka-IK-Abs-v0 \
  --livestream 2 \
  --kit_args="--/app/livestream/publicEndpointAddress=${PUBLIC_IP} --/app/livestream/port=49100" \
  --enable_cameras \
  --num_envs 1

```
