# Wph-magic
存储Wph的个人资源

## 电子书网站
https://zh.persian-books.sk/

## 快速向服务器传输文件，可能需要在同一个子网下
```
scp -r 本机目录 username@IP:远程主机目录
示例：
scp -r F:\NowData\数据增强\new wph@172.31.71.192:/media/data/workplace_wph
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

#### Ditital Ocean密码
WPH123456789SZU


### huggingface cli 下载数据集
```python
#### 依赖安装
pip install huggingface_hub[cli]        # way 1: base den install 
pip install "huggingface_hub[cli,fast_download]"    # way 2: support bigfile install
#### 配置镜像网站
export HF_ENDPOINT=https://hf-mirror.com
#### 数据集下载
huggingface-cli download --repo-type dataset --resume-download 复制数据集名称 --local-dir 下载路径
```

### tmux使用
###### 创建指定名称的会话
tmux new-session -s 会话名称
###### 切换到指定会话
tmux a -t 会话名称
###### 关闭会话
exit # 会话内输入指令
tmux kill-session -t 会话名称  # 会话外输入指令

### 单机多卡训练，关闭所有进程
```python
nvidia-smi -i 0 | grep 'python' | awk '{print $5}' | xargs -n1 kill -9
```

### 存储路径
/mnt/nvme/yfyuan/LZL/jacobian/wph

