# 项目过程中的经验与收获笔记

> 记录在构建 multi_agent_baseline 项目过程中遇到的问题、学到的知识、以及踩过的坑。
> 持续更新中。

---

## 2026-08-16:vLLM 孤儿进程 & PID 基础

### 现象

跑完 vLLM 推理后关掉了程序,但 GPU 显存仍然被占着。用 `nvidia-smi` 一看:

```
| GPU 2 | 24151MiB / 24564MiB | VLLM::Worker_TP0 |
| GPU 3 | 24151MiB / 24564MiB | VLLM::Worker_TP1 |
```

### 这是什么进程

vLLM 启动大模型推理时,如果一张显卡装不下整个模型,就把模型**拆成两半**放在两张卡上,每张卡各跑一个 Worker 进程,两个进程互相配合完成推理。

- `Worker_TP0`:拿着模型的前半截
- `Worker_TP1`:拿着模型的后半截

"TP" = Tensor Parallelism(张量并行),可以理解成"把模型劈开跑"。

### 为什么会变成孤儿进程

正常情况:主程序退出时,会亲手杀掉所有子进程,干干净净。

但 vLLM 的 Worker 是独立进程,不是主进程里的线程。如果主进程退出时没有正确通知子进程一起死,子进程就变成了孤儿:

1. 你 `Ctrl+C` 或者关掉终端 → 主进程死了
2. 主进程死之前没来得及通知子进程一起死(信号没传递到位)
3. 两个 Worker 被系统老祖宗(PID=1 的 init 进程)收养
4. 没人告诉它们该停了,就傻傻地继续占着显存

**判断方法**:用 `ps -o pid,ppid,cmd` 查看,如果 PPID(父进程 ID)= 1,说明亲爹死了,被系统收养了,这就是孤儿进程的标志。

### 什么是 PID

**PID = Process ID = 进程身份证号**

操作系统里跑的每一个程序都是一个"进程",系统给每个进程发一个唯一的数字编号,就是 PID。系统通过 PID 来找到它、管理它、杀掉它。

- `kill <PID>` → 翻译过来就是:"系统,麻烦把身份证号是 XXX 的那个进程干掉"
- PID 从 1 开始递增分配,每个新进程拿一个新号
- PID 1 是系统启动的第一个进程(init/systemd),是所有进程的祖宗

### 什么是 PPID

**PPID = Parent PID = 父进程的身份证号**

每个进程都有一个"爹"(启动它的那个进程),PPID 就是它爹的 PID。

- 正常进程的 PPID 指向启动它的程序
- 如果 PPID = 1,说明亲爹死了,被系统老祖宗收养了 → **孤儿进程**

### 杀了某个 PID 的进程后,这个编号会复用吗

**不会。** PID 只增不减,不会回头用旧的。就像银行排队取号,号发出去就不会再发给别人。极端情况下如果编号到了上限(Linux 默认几百万),系统会绕回小的编号,但只复用那些已经确认没人占用的号,而且中间隔了非常久,不会搞混。

### 坑:`kill` 杀不掉,必须用 `kill -9`

第一次执行 `kill 2309561 2309562` **没有生效**,进程还在,显存也没释放。

原因是两种信号的区别:

| 信号 | 命令 | 效果 |
|------|------|------|
| SIGTERM (15) | `kill PID` | **礼貌请求**:"请你自己退出,收拾一下再走"。进程可以选择**忽略** |
| SIGKILL (9) | `kill -9 PID` | **强制处死**:由内核直接干掉,进程**无法忽略、无法拦截** |

vLLM 的 Worker 当时阻塞在等待主进程指令的通信调用上(进程状态 `Sl`,即多线程睡眠),没有信号处理逻辑,直接把 SIGTERM 丢了。

`kill -9` 是内核绕过进程本身直接回收资源,立刻生效,显存跟着释放。

**教训:**
1. 对付赖着不走的孤儿进程,直接 `kill -9`
2. **杀完一定要验证**,别以为发了命令就成功了

### 怎么排查

```bash
# 查看 GPU 使用情况
nvidia-smi

# 只看显存数字(简洁版)
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# 查看进程的 PID、父进程、状态、已运行时长
ps -o pid,ppid,stat,user,etime,cmd -p <PID>

# 查找所有 vllm 相关进程
ps -eo pid,user,ppid,cmd | grep -i vllm

# 强制杀掉进程(推荐直接用 -9 对付孤儿进程)
kill -9 <PID>

# 验证是否真的杀掉了
ps -p <PID> || echo "已终止"
```

### 进程状态码(STAT)速查

| 状态 | 含义 |
|------|------|
| `R` | 正在运行 |
| `S` | 可中断睡眠(在等某个事件,比如等网络、等指令) |
| `D` | 不可中断睡眠(通常在等磁盘 I/O,这种 `kill -9` 都杀不掉,只能等或重启) |
| `Z` | **僵尸进程**(已经死了,但父进程没来收尸,只剩一条记录,不占资源) |
| `l` | 多线程(附加标志) |
| `s` | 会话领导者(附加标志) |

注意:我们这次遇到的是 `Sl` = **孤儿进程**(活着,占资源),**不是** `Z` 僵尸进程(死了,不占资源)。这两个概念容易混,区别是:

- **孤儿进程**:爹死了,自己还活着,继续占显存/内存 → 需要手动杀
- **僵尸进程**:自己死了,但爹没回收它的退出状态,只剩个空壳记录 → 不占资源,杀爹就没了

### 挂了 7 天,到底浪费了多少资源?

这次的孤儿进程已经跑了 **7 天 15 小时**(从 08-09 配环境那次留下的)。结论是**一半浪费了,一半没浪费**。

**浪费了:显存(占着茅坑)**

两张卡各 24GB 显存被锁死 7 天多。这期间自己想跑新实验只能用 GPU 0/1,别人一看 GPU 2/3 显存满了也直接放弃。

这是**闲置浪费**,不是消耗——像在停车场占了两个车位 7 天,车没开走,别人也停不进来。

**没浪费:算力和电(它其实在睡觉)**

看 `nvidia-smi` 的三个关键数字:

| 指标 | 数值 | 含义 |
|------|------|------|
| GPU-Util | **0%** | 完全没在算东西 |
| Perf | **P8** | 最低功耗档(P0 是全速,P8 是待机) |
| 功耗 | **6-15W** | 满载 450W,只用了 3% |

Worker 加载完模型就一直阻塞在"等指令"上,主进程死了没人发指令,所以纯粹在干等。**没有做任何计算。**

电费估算:2 张卡 × 约 10W × 168 小时 ≈ **3.4 度电**,几块钱,可忽略。

**类比**:租了个仓库堆满货(显存占用),但 7 天一次都没去动过货(算力)。仓库租金白付了(损失),但没花搬货的人工费(算力/电费没消耗)。

**一句话**:7 天里它一直在睡觉,没烧算力也没烧电,唯一损失是两张卡显存被白占了 7 天。

### 习惯:关掉推理后顺手看一眼显存

这次拖了 7 天才发现,根本原因是**关程序后没验证**。以后固定动作:

```bash
# 关掉 vLLM 之后立刻检查
nvidia-smi --query-gpu=index,memory.used --format=csv
```

看到显存降回几十 MiB 才算真的清干净了。养成习惯就不会拖一周。

### 一句话总结

**vLLM 主进程死的时候没把子进程带走,子进程变成孤儿继续占着显存。排查看 PPID 是不是 1,清理直接 `kill -9`,然后验证。它睡觉不烧电,但白占显存让别人用不了——关程序后顺手 `nvidia-smi` 确认一下。**

---

## 2026-08-16:HuggingFace 模型下载 & 缓存管理

### 场景

需要在 `/data/lsl/cache/huggingface`(数据盘)下载 Qwen3.5-9B 模型,并保证所有缓存、Python 包都落在数据盘,不污染系统盘。

### 环境配置

本机的环境变量统一在 `/data/lsl/env.sh` 中管理,最终配置如下:

```bash
export BASE=/data/lsl
export CACHE=$BASE/cache

# HuggingFace
export HF_HOME=$CACHE/huggingface          # 模型缓存根目录
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像加速
export HF_TOKEN=hf_xxx                     # 访问令牌
export TRANSFORMERS_CACHE=$HF_HOME         # transformers 库缓存
export HF_DATASETS_CACHE=$CACHE/datasets   # 数据集缓存

# 关键:XDG 缓存总开关,所有 XDG 兼容工具(pip、huggingface 等)都遵循它
export XDG_CACHE_HOME=$CACHE

# 各工具缓存
export PIP_CACHE_DIR=$CACHE/pip
export TORCH_HOME=$CACHE/torch
export UV_CACHE_DIR=$CACHE/uv
```

其中 `XDG_CACHE_HOME` 是**总开关**——Linux 下几乎所有工具默认都把缓存写到 `~/.cache`,设置这个变量后全部重定向到数据盘。`PIP_CACHE_DIR`、`TORCH_HOME` 等是各工具的专属补充。

### 下载模型:huggingface-cli

```bash
# 安装 huggingface_hub(提供 huggingface-cli 命令)
pip3 install --user huggingface_hub

# 下载模型
huggingface-cli download Qwen/Qwen3.5-9B
```

- `HF_HOME` 已指向 `/data/lsl/cache/huggingface`,模型自动落数据盘
- `HF_ENDPOINT` 已设镜像,下载不走国际流量
- 9B 模型约 19GB,4 个 safetensors 分片

### 缓存布局:新版 vs 旧版

`huggingface_hub` 不同版本的缓存目录结构不同:

| 版本 | 路径 | 说明 |
|------|------|------|
| 旧版 | `HF_HOME/models--org--model/` | 直接在 HF_HOME 下 |
| 新版(0.36+) | `HF_HOME/hub/models--org--model/` | 多一层 `hub/`,分类更清晰 |

内部结构相同(`blobs/`、`refs/`、`snapshots/`),只是外层多了一层 `hub/`。如果目录里两种布局混在一起,直接 `mv` 到 `hub/` 下统一即可,同一磁盘上移动是瞬间完成的。

### 迁移系统盘缓存到数据盘

执行后发现 `~/.cache` 在系统盘已经攒了 12GB(pip 5.1G + uv 6.6G + 其他):

```bash
# 合并迁移
rsync -a --remove-source-files ~/.cache/pip/ /data/lsl/cache/pip/
rsync -a --remove-source-files ~/.cache/uv/ /data/lsl/cache/uv/
# ... 其他目录同理
```

迁移后 `~/.cache` 只剩 4KB 空目录,`/data/lsl/cache` 共 33GB(含模型)。

### 使用项目 venv 测试模型

项目已有 venv 在 `.venv/`,内含 `transformers`、`torch`、`vllm` 等全套依赖:

```bash
source /data/lsl/env.sh                          # 加载环境变量
source .venv/bin/activate                        # 激活项目 venv

python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'Qwen/Qwen3.5-9B'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map='auto'           # 注意:需要 pip install accelerate
)

inputs = tokenizer('你好,请用一句话介绍你自己', return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# 输出:你好!我是一个由阿里巴巴集团旗下的通义实验室自主研发的...
"
```

### 踩过的坑

1. **`torch_dtype` 已废弃**:新版 transformers 要用 `dtype=torch.float16`,不是 `torch_dtype`
2. **`device_map='auto'` 需要 `accelerate`**:否则报错 `ValueError: Using a device_map ... requires accelerate`
3. **venv 里没有 `pip` 只有 `pip3`**:直接用 `python3 -m pip install` 更可靠
4. **`pip install --user` 和 `--target` 的区别**:`--user` 装到 `~/.local`(系统盘),`--target` 可以指定任意路径。如果不想系统盘有任何 Python 包,用项目 venv

### 落盘位置总结

| 内容 | 路径 | 磁盘 |
|------|------|------|
| 模型权重 | `/data/lsl/cache/huggingface/hub/` | 数据盘 |
| pip 缓存 | `/data/lsl/cache/pip/` | 数据盘 |
| uv 缓存 | `/data/lsl/cache/uv/` | 数据盘 |
| torch 缓存 | `/data/lsl/cache/torch/` | 数据盘 |
| 数据集缓存 | `/data/lsl/cache/datasets/` | 数据盘 |
| Python 包(venv) | `.venv/`(项目目录下) | 数据盘 |

---

## 2026-09-17:Self-Consistency 投票里的两个隐蔽 bug

### 背景

拿到另一份从零实现同一需求的代码(`multi_agent_baseline_0`)对照检查时,发现我们的多数投票函数 `majority_vote` 是直接对 `extract_answer` 提取出的原始答案字符串计票,没做任何归一化。修这个问题的过程中,自己又手动引入了第二个更隐蔽的 bug,顺手也记一下。

### 问题 1:等价答案被当成不同选项,票数被稀释

MATH 数据集的答案常有多种等价写法。同一个"二分之一",模型 5 次采样可能分别写成:

```text
Sample 1: 1/2
Sample 2: \frac{1}{2}
Sample 3: 0.5
Sample 4: 1/2
Sample 5: 3
```

人眼一看就知道前 4 次里有 3 次都是"二分之一"这个答案,该以 3 票胜出。但如果直接对字符串计票,`"1/2"`、`"\frac{1}{2}"`、`"0.5"` 是三个不同的字符串,分别只算 1 票、1 票、1 票,加上另一次 `"1/2"` 也只凑到 2 票——最高票变成 `1/2` 的 2 票,而不是应该聚合出的 3 票。样本小、答案形式单一时看不出问题,但样本一多、模型输出格式一杂,就可能把本该获胜的答案错误地拆没了,直接影响 Self-Consistency 的最终准确率。

**解决**:投票前先用 `normalize_answer()`(项目里为了 `is_equiv` 判等已经写好的归一化函数)统一格式,再按归一化结果分组计票。但胜出后存储/展示的仍是**原始文本**,不是归一化后的中间形式——归一化函数是为了"比较"设计的,中间产物可能很难看(比如 `\frac{1}{2}` 归一化后可能变成不完整的 `frac{1}{2`),直接存进结果文件会让人看不懂。

### 问题 2:用原始字符串当 dict key,会悄悄丢失重复次数

修问题 1 时,第一版写法是:

```python
norm_of = {a: normalize_answer(a) for a in valid}   # key 是原始答案字符串
counts = Counter(norm_of.values())
```

看起来没问题,但如果 5 次采样里有两次输出一模一样的原始字符串,比如:

```text
answers = ["3", "5", "3", "5", "7"]
```

字典的 key 不允许重复,第二个 `"3"` 会直接覆盖第一个 `"3"` 对应的条目。这个字典推导式跑完只剩 3 个 key(`"3"`, `"5"`, `"7"`),"3 出现了两次""5 出现了两次"这个信息在覆盖的那一刻就无声丢失了。后面 `Counter(norm_of.values())` 数出来的票数全变成 1 票,平票判断和最终赢家都可能算错——而且过程中**不会报任何错误**,程序照常跑完,结果却是错的。

**解决**:改成对列表(而不是字典)做批量归一化:

```python
normalized = [normalize_answer(a) for a in valid]   # 列表,允许重复元素共存
counts = Counter(normalized)
winner_idx = next(i for i, n in enumerate(normalized) if counts[n] == max_count)
winner = valid[winner_idx]                          # 用下标关联回原始答案
```

修完用 `["3", "5", "3", "5", "7"]` 验证,正确识别出 `3`/`5` 各 2 票的平票。

### 通用教训

**用字典推导式 `{key_expr: value_expr for x in items}` 做"批量转换"时,如果 `key_expr` 可能重复,重复项会被无声地合并/覆盖,不会报错,只会安静地丢数据。** 这次踩坑的场景是:需要"对列表里每一项做一次转换,同时保留每一项的原始信息用来对照",这种场景该用列表(或 `zip`)存中间结果,不能用字典——用字典的前提是 key 本身就该是唯一的,一旦借用"业务数据"当 key,就要先确认业务数据允不允许重复。

另外一层教训是关于流程本身:**修一个 bug 时新写的代码,同样需要针对边界情况专门测一遍,不能因为是"顺手改的小修复"就跳过验证。** 这个 dict-key bug 不是原有代码的问题,是修另一个 bug 时自己手动写出来的,如果没有专门用"重复答案"这个边界场景去测,它会一直隐藏在"看起来对但偶尔错"的状态里,比原来那个更难发现。

---

## 2026-09-17:LLM 构造中途崩溃也会留下孤儿进程,补一层 psutil 防护

### 现象

跑实验时发现 GPU 0/1 显存也被占满了,和 08-16 那次一模一样的症状——`nvidia-smi --query-compute-apps` 一查,又是两个 `VLLM::Worker_TP0` / `Worker_TP1`,`ps` 确认 `PPID=1`,已经是孤儿进程,活了约 21 分钟。直接 `kill -9` 两个 PID,`nvidia-smi` 复查显存降到 14MiB,清干净了。

### 排查:这次为什么"防护"没生效

项目里 `scripts/run_experiment.py` 和 `run_sanity_check.py` 早就写了 `try/finally: runner.shutdown()`,`ModelRunner.shutdown()` 也确实会清理 `self.llm` 并释放显存(见 08-16 那次笔记)。看起来该防住的,为什么还是漏了?

问题出在**构造阶段本身**,不在运行阶段:

```python
runner = ModelRunner(...)   # ← 如果这一行内部崩了,runner 根本没被赋值
try:
    ...                      # ← 这个 try 块永远不会被进入
finally:
    runner.shutdown()        # ← 更不会被执行
```

`ModelRunner.__init__` 内部调用 `LLM(...)` 时,vLLM 会先把两个 Worker 子进程拉起来,再做后续初始化(比如 sampler warmup)。如果 warmup 阶段发生 CUDA OOM 等异常,`LLM(...)` 这次调用本身失败,异常直接从 `ModelRunner(...)` 这一行往外抛——**子进程已经真实存在了,但 Python 这边连 `self.llm` 都没赋值成功**,更别提 `runner` 变量。调用方的 `try/finally` 从头到尾都没被进入过,`shutdown()` 自然管不到已经拉起来的子进程。

一句话:**已有的 `try/finally` 只保护"构造成功之后"的阶段,保护不了"构造过程本身"这段时间冒出来的子进程。**

### 解决:在 `__init__` 内部自己包一层

在 `ModelRunner.__init__` 里,把 `LLM(...)` 调用本身包进 `try/except`,用 `psutil` 记录调用前后的子进程差异,构造失败时只杀"新冒出来的"那些,再把原始异常重新抛出去(不能吞掉,调用方还得知道构造失败了):

```python
before_children = set(psutil.Process().children(recursive=True))
try:
    self.llm = LLM(...)
except Exception:
    after_children = set(psutil.Process().children(recursive=True))
    for proc in after_children - before_children:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    raise
```

用**前后差集**而不是直接杀"当前所有子进程",是为了不误杀构造前就已经存在、跟这次 `LLM(...)` 调用无关的其他子进程。

### 加依赖时的环境小坑:系统没有 `pip`

这台机器系统 Python (`/usr/bin/python3`) 没装 `pip`(`No module named pip`),项目用的是 `uv` 管理依赖(`pyproject.toml` + `uv.lock` + `.venv`)。一开始尝试全局装 `psutil` 走不通,改用项目自己的包管理方式:

```bash
uv add psutil     # 自动做三件事:
                  # 1. 更新 pyproject.toml 的 dependencies
                  # 2. 更新 uv.lock
                  # 3. 装进 .venv
```

同时手动把 `psutil>=5.9.0` 补进 `requirements.txt`,保持两份依赖声明一致(项目里 `pyproject.toml` 是 uv 的权威来源,`requirements.txt` 是给不用 uv 的场景兜底,两者都要维护)。

**教训:环境里没 `pip` 不代表装不了包,先看项目用的是什么包管理工具(`pyproject.toml`/`uv.lock` 一般对应 `uv`),用它自带的命令加依赖,别绕去装全局 pip。**

### 顺带一提:这个文件里 `import psutil` 破了个既有约定

`model_runner.py` 开头原本特意写了"本文件在无 GPU/vLLM 的环境下不可运行",配合 `vllm`、`SamplingParams` 都是**延迟导入**(写在函数体内,而不是文件顶部)——这样即使本机没装 vLLM,也能正常 `import src.model_runner` 做别的事(比如跑 dataclass 相关的单测)。

这次新加的 `import psutil` 图省事写在了文件顶部,没跟着这个约定延迟导入。目前问题不大,因为 `psutil` 是纯 Python 小依赖,装起来没有 vLLM 那么重、那么容易失败,而且已经加进了 `pyproject.toml`,正常走 `uv sync` 就会有。但严格来说这是对原有"本文件应可在无 GPU 环境下 import"设计意图的一个小偏离,如果以后要在完全没网、没装 `psutil` 的极简环境里 `import` 这个模块跑单测,会在这一行直接炸掉。暂时先记录,没有改动。

### 验证:写测试时踩的一个小坑——kill 完不是"没了",是"变成 zombie"

写测试验证"构造失败时新子进程会被杀掉",第一版断言直接检查 `psutil.Process().children(recursive=True) == []`,结果跑起来失败:子进程明明被 `kill()` 了,`children()` 里却还能看到它,状态是 `zombie`。

这不是 bug,是正常现象(参考 08-16 笔记里 Z 状态的定义):**进程被杀死之后,如果父进程没有调用 `wait()` 去"收尸",子进程会短暂停留在僵尸状态**——它已经死了,不占显存/内存/CPU,只是进程表里还留着一条退出状态的记录,等父进程收尸后才会彻底消失。测试跑在 pytest 进程里,没人主动 `wait()`,所以子进程短时间内一直是 zombie。

**修正**:断言不应该要求"完全看不到这个进程",而应该是"这个进程不再是活着占资源的状态":

```python
survivors = [
    p for p in psutil.Process().children(recursive=True)
    if p.status() != psutil.STATUS_ZOMBIE
]
assert survivors == []
```

**教训:验证"进程被杀掉了"时,别用"进程列表里还能不能查到它"做判断标准,准确的判断是"它是不是 zombie 状态"——同一个坑在真实事故排查(08-16)和写单测验证时都会踩一次,值得多留意。**

---

## 2026-09-17:vLLM 启动阶段 OOM——背景知识:引擎启动到底在干什么

### 背景:为什么"跑推理之前"也能 OOM

一个容易搞错的直觉是:显存爆掉肯定是因为"塞了太多请求/太长的文本"。但这次的 OOM 发生在**跑我们的 5 道题之前**——`LLM(...)` 这一行还没返回,程序还没机会处理任何一道题就先炸了。要理解为什么,得先知道 vLLM 的 `LLM(...)` 构造函数内部不是"读完权重就直接能用",而是要走完一整套准备流程。按实际日志顺序过一遍:

**第 1 步:解析配置,决定"怎么切"**

```
Resolved architecture: Qwen3_5ForConditionalGeneration
Using max model len 8192
```

只是根据 `model_path`、`tensor_parallel_size=2`、`max_model_len=8192` 等参数识别模型架构、规划切分方式,还没真正碰显存。

**第 2 步:启动 Worker 子进程**

```
(EngineCore pid=317937) Initializing a V1 LLM engine ...
(Worker pid=318148) world_size=2 rank=0 ...
(Worker pid=318149) world_size=2 rank=1 ...
```

`tensor_parallel_size=2` 意味着模型被"劈成两半",由两个独立的 Worker 子进程分别管理 GPU 0 和 GPU 1,两者之间用 NCCL(显卡间通信库)互相传数据——这就是 08-16 笔记和上一条笔记里反复出现的 `Worker_TP0`/`Worker_TP1`,子进程一旦拉起来,主进程构造过程中途崩了它们也不会自动跟着退出。

**第 3 步:把模型权重搬进显存(这是"地基")**

```
Loading safetensors checkpoint shards: 100% ...
Model loading took 111.41 seconds
Model loading took 8.91 GiB memory
```

把硬盘上的 safetensors 权重文件读进 GPU 显存,9B 模型这一步就吃掉每张卡 8.91GB,耗时接近 2 分钟。这是后面所有步骤都要在剩余显存里"抠"空间的起点。

**第 4 步:`torch.compile` 编译计算图**

```
Compiling a graph for compile range (1, 8192) takes 28.64 s
```

把模型的计算过程编译成更高效的执行形式(类似把手写步骤变成流水线标准作业),主要费时间,不太占显存。

**第 5 步:KV Cache 探测与分配(这次事故的根源)——见下一节详细展开**

**第 6 步:捕获 CUDA Graph**

```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|...
Graph capturing finished in 6 secs, took 3.50 GiB
```

CUDA Graph 是另一种加速手段:把一套固定的 GPU 计算步骤"录成一段回放脚本",以后重复执行时不用每次都重新下发指令,直接"放录像"更快。这一步额外占用 3.5GB。

**第 7 步(炸的这一步):热身采样器**

```
ERROR ... torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 244.00 MiB.
RuntimeError: CUDA out of memory occurred when warming up sampler with 256 dummy requests.
```

vLLM 最后要测试一次"从模型输出的概率分布里随机抽样"这个环节能不能正常工作,做法是假装同时来了 256 个请求跑一遍最坏情况(这个 256 是 `max_num_seqs` 的默认值,**跟我们实际只有 5 道题完全无关**)。这一步本身占用不大,但前面几步已经把显存挤得只剩几十 MB,连 244MB 都分不出来,直接崩溃。

**一句话串起来**:`LLM(...)` 构造 = 解析配置 → 拉起 Worker 子进程 → 权重搬进显存 → 编译计算图 → 探测并分配 KV Cache → 捕获 CUDA Graph 录像 → 用一批"假请求"做最终热身测试。我们这次是在倒数第二步(KV Cache 探测)埋下隐患,在最后一步(256 路假请求热身)被压垫穿而炸掉,跟三种推理方法自己的代码逻辑完全没关系,是引擎启动阶段自己的问题。

### 排查:日志里的 "Overriding num_gpu_blocks=0 with num_gpu_blocks_override=512" 是怎么回事

先解释 KV Cache 是什么:模型跟你对话/回答时,不用每次都从头重新理解前面说过的话,而是把算过的中间结果("每个正在处理的请求,历史上下文对应的注意力缓存")存起来复用,后续生成才能快。这块专门存中间结果的显存区域就是 KV Cache,以"块"(block)为单位分配。

第 5 步的日志:

```
Initial profiling/warmup run took 93.51 s
Overriding num_gpu_blocks=0 with num_gpu_blocks_override=512
```

vLLM 会先做一次"探测":假装塞满数据,看权重 + 编译产物占掉显存后还剩多少空间能用来做 KV Cache,算出**理论上能分配的块数**。这次探测结果是 **0 块**——也就是 vLLM 自己判断"权重 + 其他开销已经把显存吃得差不多了,不该再挤 KV Cache 进去了"。但紧接着有个"override"(强制覆盖)把这个 0 硬改成了 **512 块**,不管三七二十一先分配再说。

**这里最初的猜测(以为是服务器环境里残留的 `VLLM_NUM_GPU_BLOCKS_OVERRIDE` 之类的环境变量在起作用)排查后被排除了**——检查了当前 shell 的 `env`、项目的 `env.sh`、`~/.bashrc`/`~/.profile`、`/etc/environment`、bash history,以及日志里记录的实际启动命令,哪里都没有这个变量或对应的 `--num-gpu-blocks-override` 命令行参数。真正的原因在 vLLM **自己的源码逻辑**里,跟外部环境无关:

1. vLLM 内部有个参数叫 `max_cudagraph_capture_size`(CUDA Graph 最大捕获尺寸,对应第 6 步),没手动指定时按公式自动算:`min(max_num_seqs * 2, 512)`。我们当时用的是 `max_num_seqs` 默认值 **256**,算出来正好是 `min(256*2, 512) = 512`。
2. 在正式做 KV Cache 探测之前,vLLM 有个专门的准备步骤(`_init_minimal_kv_cache_for_profiling`),目的是先分配一块"最小够用"的 KV Cache,好让后面第 6 步捕获 CUDA Graph 时有地方可用。它的做法是:**临时**把 `num_gpu_blocks_override` 设成刚才算出的 512,用 `available_memory=0` 跑一次分配函数(这一步故意让"理论可用显存"算成 0,只是为了走一遍分配流程),算完立刻把这个临时值恢复原状。
3. 这个"临时赋值"恰好命中了 vLLM 另一处日志逻辑(`may_override_num_blocks`):只要 `num_gpu_blocks_override` 不是 `None`,就打印 `Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d`,并真的按这个数字分配。于是日志里就看到了"探测算出 0,却被覆盖成 512"这行——**这不是外部注入的覆盖,是 vLLM 自己为了给 CUDA Graph 准备最小缓存,临时借用了同一个开关**。

串起来的因果链:`max_num_seqs=256`(默认值)→ 算出 `max_cudagraph_capture_size=512` → 第 5 步临时强制分配 512 块 KV Cache(实测占用 9.52GB)→ 权重 8.91GB + CUDA Graph 池 3.5GB + 这 9.52GB ≈ 23.58GB,顶穿了 `gpu_memory_utilization=0.85` 原本给的预算(~20.1GB)→ 第 7 步用 256 路假请求做最终热身时,只剩不到 50MB,连 244MB 都分不出来,`torch.OutOfMemoryError`。

**教训:日志里出现"override"字样,第一反应是查外部配置(环境变量/CLI flag)当然合理,但如果哪儿都查不到,要往框架自己的内部实现去看——很多"override"其实是框架内部两个子步骤之间借用同一个字段传参,不是真的被外部覆盖。**

### 解决:把 `max_num_seqs` 从默认 256 调到 32

`max_num_seqs` 是 vLLM 里"引擎同时能处理的最大并发请求数"这个参数,默认 256 是给通用在线服务场景设计的(要能接住大量并发用户)。而这个项目一次实验最多 200 道题、Self-Consistency 用 `n=5` 采样,**从来不需要 256 路并发**,这个默认值纯粹是"引擎按通用场景配的安全上限",跟我们实际的工作负载不匹配。

把它调低能同时压低两处启动阶段的显存开销:

- 第 5 步的 `max_cudagraph_capture_size = min(max_num_seqs*2, 512)` 会跟着变小,`max_num_seqs=32` 时算出 `min(32*2, 512)=64`,临时分配的最小 KV Cache 块数从 512 降到 64,显存占用同比例大幅下降。
- 第 7 步的最终热身也会用 32 路假请求测试,而不是 256 路,进一步降低热身阶段本身的瞬时显存需求。

改动位置:

```python
# src/config.py
@dataclass
class ModelConfig:
    ...
    max_num_seqs: int = 32

# src/model_runner.py
class ModelRunner:
    def __init__(self, ..., max_num_seqs: int = 32) -> None:
        ...
        self.llm = LLM(..., max_num_seqs=max_num_seqs)
```

两处调用方(`scripts/run_sanity_check.py`、`scripts/run_experiment.py`)构造 `ModelRunner` 时都要补上 `max_num_seqs=model_config.max_num_seqs`,否则 `ModelConfig` 里加的字段不会真正传到 `LLM(...)`。

### 一句话总结

**vLLM 的 `LLM(...)` 构造要走完"拉子进程 → 加载权重 → 编译 → 探测/分配 KV Cache → 捕获 CUDA Graph → 假请求热身"一整套流程才算真正就位,任何一步都能单独把显存挤爆,跟业务代码是否已经开始跑无关。日志里的"override"不一定来自外部环境变量,也可能是框架内部两个准备步骤在互相借用同一个配置字段——查不到外部来源时,去翻框架源码里这个字段还被谁写过。这次的根源是默认 `max_num_seqs=256` 按通用在线服务场景设的安全上限,跟我们几十道题的批量实验场景不匹配,调小到实际需要的量级(32)就同时压低了两处启动阶段的显存峰值。**

---

## 2026-09-18:阶段性复盘——这几次问题为什么当时没想到,以后笔记该多记点什么

### 背景

笔记记到这里,已经踩过孤儿进程(两次)、投票统计 bug(两次)、启动阶段 OOM 三类问题。单独看每一条都已经在笔记里分析过"根本原因",但还没有回头看一遍"这些问题当时为什么没能提前想到"——如果能找到共性,以后写代码/记笔记时就能主动堵一部分,而不是每次都靠事故复盘。

### 逐条复盘:当时的思路卡在哪一步

| 问题 | 当时的假设/思路 | 假设错在哪 |
|---|---|---|
| 08-16 孤儿进程(关程序后显存不释放) | "关掉程序 = 进程和它的子进程都没了" | 没意识到 vLLM 的 Worker 是独立子进程,主进程退出不保证子进程跟着退出;把"逻辑上的一个程序"和"操作系统里的进程树"混为一谈 |
| 09-17 投票 bug(等价答案分票 / dict key 覆盖) | "`extract_answer` 提取出字符串就能直接比较/计票" | 已经写了 `normalize_answer` 专门给 `is_equiv` 判等用,但没意识到"计票"和"判等"是同一类需求(判断两个答案是不是"同一个"),新写 `majority_vote` 时没复用已有工具,是"重复发明"而非"利用已有能力" |
| 09-17 构造中途崩溃留孤儿进程 | "`try/finally` 包住 `runner = ModelRunner(...)` 之后的代码就够了" | 隐含假设"这一行只有两种结果:成功返回,或者失败但没产生任何需要清理的副作用",没考虑到"失败,但已经产生副作用(子进程)"这个中间状态 |
| 09-17 启动阶段 OOM | "显存问题应该出现在处理请求的时候" | 把"程序还没跑我的业务逻辑"等同于"不可能出问题",没考虑到框架自己的初始化流程(热身、探测)本身就要占显存,跟业务代码是否已经跑起来无关 |

### 共性:三种"想不到"的模式

把上面几条抽象一下,当时想不到的原因基本落在三类:

1. **把"抽象层"当成了"物理层"的替身**——以为"程序"是一个不可分的整体(08-16 的进程树、09-17 的构造函数),但实际执行时它由多个独立的部分组成,各部分有自己的生命周期,可能不同步。
2. **同一个语义需求,在代码里长了两次"腿"**——`normalize_answer` 本该覆盖"判断两个答案是否算同一个"这整类问题,但新写 `majority_vote` 时没意识到自己在解决同一类问题,写了第二套(有 bug 的)逻辑。这种重复不会报错,只会在测试没覆盖到的场景里悄悄错。
3. **只验证了"正常路径",没验证"中间状态"**——`try/finally`、`shutdown()` 这些防护逻辑第一次写的时候,思路停在"成功"和"失败"这两个终态,没去想"失败发生在流程的哪一步"会不会影响防护本身能不能生效。

这三类背后其实是同一件事:**写代码时脑子里的"模型"比实际系统的"模型"更简单**,简化掉的那部分细节,不出事就一直看不见,出事才暴露。

### 以后写笔记/写代码时,主动检查清单

根据上面的复盘,以后遇到类似场景,可以主动问自己这几个问题,而不是等出错了再补:

1. **写"资源生命周期管理"代码(进程、连接、显存、文件锁)时**:主动画一下"这个资源从创建到释放,中间有几个可能失败的节点",分别检查每个节点失败时,防护逻辑是不是还能覆盖到——不能只看"整体成功/整体失败"这两种情况。
2. **写"判断两个值是否算同一个"这类工具函数(如 `normalize_answer`)时**:顺手在函数注释/笔记里列一下"这个函数应该被用在哪些地方"(判等、去重、计票、排序……),以后新写涉及"比较/分组"逻辑的代码时,先检查这份清单里有没有能直接复用的,而不是凭感觉重新写。
3. **用字典/集合做"批量转换或去重"时**:先问一句"参与运算的 key 允许重复吗?如果重复,信息会不会被覆盖掉?"——这类 bug 的共同特征是"不报错,只丢数据",很难在跑起来的时候自己发现,必须在写的时候主动想。
4. **遇到框架/依赖的默认配置值报错或行为异常时**:先查这个默认值是为什么场景设计的(通用/极限场景),再对比自己的实际场景是不是符合,而不是第一反应就去改配置数值本身——数值改对了,但如果没搞懂为什么默认值不适配,下次换个参数组合可能又踩到类似的坑。
5. **笔记里除了记录"最终结论",也记录"排查过程本身排除了什么"**:比如这次 OOM 排查记录了"查过环境变量、CLI 参数,都排除了",这份"排除清单"本身就是有效信息——下次遇到类似"看起来被外部覆盖"的现象,可以直接跳过这些已经验证过没问题的方向。
6. **如果有条件对照另一份独立实现(就是这次怎么发现 09-17 投票 bug 的)**:这类"语义正确性"bug(代码能跑、测试能过,但逻辑本身有偏差)往往不是靠自己反复读代码发现的,而是靠"换一个人从零写一遍,再对比差异"发现的——如果没有第二份实现可以对照,至少要针对"业务语义的边界情况"(等价形式、重复值、平票)专门设计测试用例,不能只测"跑不跑得通"。

### 一句话总结

**这几次问题当时想不到,根本原因是脑子里对系统的简化模型比实际系统少考虑了"中间状态"(进程树的独立生命周期、构造函数内部的失败时机、字典 key 覆盖的隐蔽性、框架初始化阶段本身的显存开销)。以后主动检查资源生命周期的中间节点、复用已有的"比较类"工具函数、对字典/集合的 key 唯一性打个问号、记录排查时排除掉的方向、并对业务语义的边界情况专门测试,能在同类问题变成事故之前先堵掉一部分。**

---

## 2026-09-19:加 `enable_thinking` CLI 开关 + 把 `max_model_len`/`max_tokens` 从 8192/4096 调大

### 背景

两个独立但顺带一起做的改动:

1. CoT / Self-Consistency 一直靠 `enable_thinking=None`(不传参数,由模型自己决定要不要思考)跑,想加个 CLI 开关能显式控制,方便对比"强制开思考" vs "强制关思考"的效果。
2. 复盘实验失败样本时,发现有一部分失败原因是 `max_tokens=4096` 把思考过程截断了(`finish_reason == "length"`),而不是模型真的答错——这类"因为长度限制判错"混在"真答错"里,会污染 accuracy 统计,想把长度限制放宽到基本不会成为瓶颈。

### 改动 1:`enable_thinking` 透传到 CLI

`run_single_method`(`src/methods/base.py`)本来就支持 `enable_thinking` 参数,但只有 `run_direct` 在用(固定传 `False`),`run_cot`/`run_self_consistency`/`scripts/run_experiment.py` 都没把这个参数暴露出来。加了:

- `run_cot`、`run_self_consistency` 新增 `enable_thinking: bool | None = None` 形参,直接转发给 `run_single_method`。
- `run_direct` 也接收这个形参(保持三个方法调用签名一致,方便 `run_experiment.py` 统一传参),但内部 `del enable_thinking` 忽略掉,始终强制 `enable_thinking=False`——direct 方法的定义就是不思考,不应该被 CLI 开关意外打开。
- `run_experiment.py` 加 `--enable-thinking`/`--disable-thinking` 互斥参数(`argparse` 的 `mutually_exclusive_group`),默认都不传时 `enable_thinking=None`,保持原来"模型自己决定"的行为。

**没踩坑,但值得记录的设计选择**:`run_direct` 接收但忽略参数,而不是从签名里干脆去掉——如果去掉,`run_experiment.py` 里 `run_fn(...)` 那处调用就得对 `direct` 特殊处理(if method == "direct": 不传 enable_thinking),三个方法的调用方式不统一,增加调用方的分支复杂度。用"统一签名,内部按需忽略"换掉"调用方按方法分支",复杂度从调用方转移到实现方,更符合"三个方法应该有一致接口"这个既有设计(`METHOD_RUNNERS` 就是当同类接口用的)。

### 改动 2:`max_model_len` 8192→16384,`max_tokens` 4096→13000

先查了这俩数字当年是怎么定的:8192 不是模型能力上限,是 09-17 笔记记录的那次启动阶段 OOM 调参时(配合 `max_num_seqs=32`、`gpu_memory_utilization=0.85`)试出来的"能跑起来"的值,从没查过模型实际支持多长。这次去读了模型目录下的 `config.json`:

```
"max_position_embeddings": 262144
```

Qwen3.5-9B 实际支持 262144,8192 只是当时保守设的显存预算,离模型上限差了 32 倍。另外顺手确认了这个模型的架构对长上下文比较友好:`layer_types` 里 32 层只有每第 4 层是 `full_attention`(KV Cache 随长度线性增长),其余都是 `linear_attention`(状态大小基本不随长度增长)——同样调大 `max_model_len`,这个模型的显存增量比标准全 attention 模型小。

但"友好"不等于"零代价":09-17 笔记分析过,启动阶段的 KV Cache 探测范围、CUDA Graph 编译范围(`Compiling a graph for compile range (1, N)`)都是按 `max_model_len` 算的,调大了这两处开销也会跟着变大,有可能在启动阶段重新撞上那次的 OOM。所以没有直接冲到 262144,选了一个折中的 16384(2 倍),`max_tokens` 跟着调到 13000(留出输入长度的余量),而不是"设成理论最大值"——这是"改配置前先搞懂默认值为什么这么设"(09-18 笔记里第 4 条检查清单)的直接应用。

**这次改动还没验证是否会在启动阶段重新触发显存问题,需要在服务器 GPU 环境跑一次 `run_sanity_check.py` 确认。**

### 一句话总结

**改 CLI 参数时,统一接口比调用方分支更值得优先,即使意味着某个实现要接收一个自己不用的参数。改显存相关的配置数值前,先查这个数值当年是"按模型能力上限设的"还是"按当时的显存预算试出来的"——8192 属于后者,而且从没验证过模型真实上限,这次翻 `config.json` 才发现只用了模型能力的 1/32。放大配置时选比当前值大但离理论上限有安全边际的折中值(16384 而非 262144),把"验证会不会重新 OOM"的风险留在可控范围内,而不是一步冲到底再排查一次完整的 OOM 事故。**

---

## 2026-09-19:q161 判分错误——模型答对了但被判错,两层原因叠在一起

### 现象

sanity check 结果里,`cot`/`run_seed42` 的 q161 被判为错误:`prediction: "1"`,`ground_truth: "14"`。但翻开原始输出 `raw_outputs/q161.txt` 一看,模型明明已经在文本靠前的位置正确推导并写出了 `\boxed{14}`——答案本身没错,是判分环节把它判错了。

### 排查:两层原因,缺一都不会出这个 bug

**第一层(模型侧):正确给出答案后没有停下来,陷入重复循环直到被截断**

q161 的原始输出在写出 `\boxed{14}` 之后,并没有输出结束符(EOS)收尾,而是不断重复输出同样的 `\boxed{14}` 片段,一直循环到 `max_tokens` 上限,最后一次重复恰好在写到 `\boxed{1` 时被硬截断(`finish_reason == "length"`),缺右花括号,是一个不完整、不闭合的片段。

**第二层(提取侧):`last_boxed_only_string` 只看"最后一次出现",不闭合就直接放弃**

`src/extract_answer.py` 里原来的 `last_boxed_only_string` 逻辑是:找文本里最后一个 `\boxed`/`\fbox` 出现的位置,从那里往后数花括号配对,取配对内容。但因为最后一次重复被截断,配对天生就凑不齐,函数直接返回 `None`,完全没有考虑"再往前找上一个完整的 `\boxed{}`"这个选项。`extract_answer` 拿到 `None` 后按预设的兜底逻辑退化成"抓文本里最后出现的裸数字"的正则,而这个正则抓到的刚好是截断片段里孤零零的那个 `1`——于是 `prediction` 变成了 `"1"`。

**两层因果关系**:如果模型没有陷入重复循环(第一层没发生),`\boxed{14}` 就是文本里唯一/最后的完整占用,`last_boxed_only_string` 原逻辑也能对;如果提取逻辑本身支持"跳过不完整的,退回上一个完整的"(第二层不是这么写的),即使模型重复也不影响最终提取结果。两层缺一个,q161 都不会出这个错——是"模型侧的退化行为"暴露了"提取侧本来就存在但一直没被触发"的一个逻辑漏洞。

### 解决 1(已修,提取侧):`last_boxed_only_string` 改成从后往前跳过不完整的occurrence

把原来"只看最后一次出现"改成:维护一个不断收缩的搜索终点 `search_end`,每次从 `search_end` 往前找最近一个 `\boxed`/`\fbox`,尝试配对花括号;配对失败(不闭合)就把 `search_end` 收缩到这次失败位置之前,继续往前找上一个,直到找到一个能配对成功的,或者彻底找不到再返回 `None`。用一条构造出的复现场景验证:

```python
text = "Thus the answer is \\boxed{14}.\n\n\\boxed{14}\n\n\\boxed{1"
assert extract_answer(text) == "14"
```

补充了对应测试 `test_extract_boxed_truncated_repetition_falls_back_to_earlier_complete_one`(`tests/test_extract_answer.py`),连同已有 15 条测试全部通过。

### 解决 2(已加,模型侧):给采样加 `repetition_penalty`,从源头抑制退化重复循环

只修提取侧只能保证"万一模型又重复循环,不会因为截断位置不巧而判错",但没解决模型侧本身白白烧掉一整份 `max_tokens` 预算在无意义重复上的问题(既浪费 token/时间,也不能保证下次重复不会连"上一个完整的 `\boxed{}`"都卷进截断范围)。

vLLM 的 `SamplingParams` 本身就支持 `repetition_penalty`(装的 0.19.1 版本确认有这个字段,默认 1.0 即不惩罚,>1.0 时对已经出现过的 token 降低再被选中的概率),用它从生成阶段就压低"复读同一段内容"的概率,比在提取阶段补救更治本。

改动位置:

```python
# src/model_runner.py:ModelRunner.generate() 新增形参,原样传给 SamplingParams
def generate(self, prompts, temperature, max_tokens, n=1, seed=None, repetition_penalty=1.0):
    sampling_params = SamplingParams(..., repetition_penalty=repetition_penalty)

# src/config.py:MethodSamplingConfig 新增字段,三个方法配置都带上
@dataclass
class MethodSamplingConfig:
    temperature: float
    n: int = 1
    repetition_penalty: float = 1.05  # 温和惩罚,不明显影响正常回答质量
```

`src/methods/base.py` 的 `run_single_method` 以及 `direct.py`/`cot.py`/`self_consistency.py` 三个方法文件都相应加了这一参数的透传,从各自的 `MethodSamplingConfig` 一路带到 `runner.generate(...)`。默认值选了 1.05 这种温和值而不是更激进的数字,是因为惩罚一旦调太高,可能反而干扰正常解题时确实需要重复某些 token(比如公式里的变量名、多步验证时重复引用同一个数)的场景——具体这个值够不够压住 q161 这类退化循环,还没有在真实 GPU/vLLM 环境里重跑验证过,只做了参数传递路径的单测覆盖(`tests/test_model_runner.py` 新增两条,用假的 `LLM`/`SamplingParams` 验证 `repetition_penalty` 确实被传到位,不需要真实 GPU)。

### 一句话总结

**判分错误不一定是模型答错,也可能是"模型侧退化重复 + 截断" 和 "提取逻辑只看最后一次出现、不会回退" 两层问题叠加导致的假阳性——提取侧的 bug 平时隐藏得很深,只有配合特定的截断位置才会暴露。修复分两层:提取侧让 `last_boxed_only_string` 支持跳过不完整的 occurrence 回退到上一个完整的(治标,防止再次判错),模型侧加 `repetition_penalty` 从采样源头压低退化重复的概率(治本,减少这类循环真的发生)。后者的实际效果还需要在真实环境跑一次采样验证。**

---

## 2026-09-19:q161 遗留问题的后续——调大预算 + 截断时禁用兜底提取

### 现象

上一条 q161 笔记修完之后留了一个没验证的疑点:`repetition_penalty` 到底能不能压住退化重复循环,当时没在真实环境跑过。真的跑了之后发现:即使不发生 q161 那种"答案写完了还在重复"的退化循环,Level 5 难题本身在开启思考模式(`enable_thinking`)后,正常的、非重复的推理探索(尝试多种情况、多条分支)也经常长到超过当时 `max_tokens=13000` 的预算,还没写到 `\boxed{}` 就被硬截断了。这跟 q161 是同一个 `finish_reason == "length"` 现象,但原因不同:q161 是"答案已经有了,只是没停下来";这次是"答案确实还没推出来,预算给太小"。

### 排查

两种"被截断"场景需要区分对待:

1. **截断,但之前已经写过完整 `\boxed{}`**(如 q161):上一条笔记已经修好——`last_boxed_only_string` 会跳过不闭合的 occurrence,回退到前面那个完整的。
2. **截断,且全文都没有任何完整 `\boxed{}`**(这次遇到的新情况):模型确实还没推理完,`extract_answer` 原来的兜底逻辑(找不到 `\boxed{}` 就退化成"抓文本里最后一个裸数字")会硬抓一个数字凑出一个 `prediction`。但这个数字只是截断点附近偶然出现的东西,例如"情况一:乘积是33,所以 a_{n-1} 可能是1"里的"1",它是某个分支探索里的中间数字,不是模型的最终结论。把它当成一次有效预测,等于伪造了一个近似随机的答案。

第二种情况在 Self-Consistency 里危害更大:5 次采样里如果有几次是这种"伪答案",会直接参与多数投票,可能把一个错误的伪造答案投成最终 `prediction`,而不是老老实实告诉你"这道题没采出有效答案"。

### 解决 1:调大 `max_tokens`/`max_model_len`,先从源头减少截断发生的频率

`src/config.py` 里 `max_tokens` 从 13000 调到 20000,`max_model_len` 从 16384 调到 24576(留够输入 prompt + 20000 输出的空间)。这只能缓解,不能保证消除——Level 5 难题的推理长度分布本身有长尾,调多大都可能有极少数题目还是会碰到上限。

**后续验证**:调到 20000 之后重新跑 sanity check,q161 在 `cot`/`self_consistency` 里依然 `finish_reason == "length"`(self_consistency 甚至 5 个样本全部截断)。查看 `raw_outputs/q161.txt` 截断处的文本,里面是模型在老老实实穷举符号组合、验证有理根定理,没有重复段——不是 q161 那种"答案写完了还在复读"的退化循环,是这道 Level 5 题本身需要的推理长度就超过 20000。这次先不给这类题目的推理长度设限,把 `max_tokens` 进一步调到 32000、`max_model_len` 调到 36864。仍然截断的极端长尾case交给下面解决 2 的 `allow_fallback` 兜底,不再继续死磕预算数字。

### 解决 2:`extract_answer` 加 `allow_fallback` 参数,截断且无 `\boxed{}` 时禁用兜底规则

既然调大预算不能保证消除截断,提取逻辑就必须能诚实地承认"没有可靠信号时就说不知道",而不是每次都硬凑一个答案。`extract_answer(model_output, allow_fallback=True)` 的三层提取优先级:

1. `\boxed{}`/`\fbox{}`(最高置信度,模型自己明确写的最终答案)
2. "final answer is X" / "答案是X" 这类短语正则匹配
3. 文本里最后一个裸数字/分数(置信度最低)

`allow_fallback=False` 时关掉第 2、3 层,第 1 层找不到就直接返回 `None`,不再退化去抓数字。`src/methods/base.py` 的 `run_single_method` 按采样的 `finish_reason` 决定要不要允许兜底:

```python
sample_answers = [
    extract_answer(s.text, allow_fallback=s.finish_reason != "length")
    for s in out.samples
]
```

正常结束(`finish_reason == "stop"`)的采样照常允许兜底;被截断的采样禁用兜底,没写到 `\boxed{}` 就直接是 `None`。

`self_consistency.py` 的 `majority_vote` 不需要改——它已经有 `if not valid: return None, "no_valid_answer"`,`None` 会被自动过滤掉不参与投票。以前 q161 这类"全部截断且无 `\boxed{}`"的场景会投出一个类似 `"33 (1/5) [tie]"` 的误导性结果,现在会诚实地变成 `(None, "no_valid_answer")`。

新增测试覆盖:`tests/test_extract_answer.py` 两条(验证 `allow_fallback=False` 时截断场景返回 `None`、`\boxed{}` 命中不受影响),`tests/test_methods_base.py`(新文件,之前这个模块零覆盖)两条,用一个不依赖真实 vLLM 的假 `ModelRunner`(只需实现 `build_prompt`/`generate` 两个方法)验证 `run_single_method` 按 `finish_reason` 正确决定 `allow_fallback`。

### 一句话总结

**截断(`finish_reason == "length"`)分两种:答案已经有但没停下来(上一条笔记修的),和答案确实还没推出来(这条修的)。前者靠"跳过不完整 occurrence 回退到上一个完整 `\boxed{}`"解决,后者靠"没有 `\boxed{}` 就不要硬凑答案"解决——本质都是同一条原则:提取逻辑对截断样本要更谨慎,宁可诚实地返回 `None`,也不要伪造一个看似合理但其实是随机噪音的预测,尤其是在 Self-Consistency 的多数投票场景下,一个伪造答案可能直接污染最终结果。**

---

# 概念笔记(不依赖具体事故,纯知识点)

> 上面的记录都是"踩坑复盘",每条都挂在一次具体事故上。但项目里还有一类知识——写代码前就该搞懂的基础概念(比如这次的 temperature、seed)——它们不是从事故里挖出来的,硬凑一个"现象/原因/解决"的事故模板反而别扭。所以单开这一部分,专门放这种"纯知识点"笔记,跟上面的事故日志分开存放,方便以后只想复习概念时不用从事故堆里翻。

## 2026-09-18:temperature 和 seed 到底是什么,项目里为什么这么配

### 1. temperature 是什么:控制模型"选词随机性"的旋钮

模型生成文本是一个词(token)一个词往后接的。每接一个词之前,模型内部都会算出一份"候选词概率表",比如接下来这个词,"是"占 40% 概率,"的"占 25%,"了"占 10%,其余词分掉剩下的 25%。temperature 决定的是:拿到这份概率表之后,怎么从里面"抽签"选一个词。

- **temperature = 0**:不抽签,直接选表里概率最高的那个词(这次例子里就是"是")。这叫贪婪解码,完全确定——同一个输入,问一百次,一百次的输出都一字不差。
- **temperature 越大**:概率表会被"拉平"。本来只有 10% 概率的"了"也有更大机会被抽中,本来垫底的冷门词也能偶尔冒出来。同一个问题问 5 次,5 次很可能走上 5 条不同的路,答案也不一样。

一句话:temperature = 0 是"每次都选最稳的那个",temperature 越高越像"掷骰子,冷门选项也有机会"。

### 2. temperature 在这个项目里怎么配的([[project-config]] 见 `src/config.py`)

- **Self-Consistency 必须用非零 temperature**(项目定的是 0.7):这个方法的核心是"同一道题采样 5 次,少数服从多数投票"。如果 temperature=0,5 次采样会因为贪婪解码完全走同一条路,得到 5 个一模一样的答案——投票就变成了"1 个答案自己跟自己比",没有任何意义。必须靠非零 temperature 让 5 次真的产生分歧,投票才有价值。
- **Direct 和 CoT 一开始的直觉是用 temperature=0**(反正只采样 1 次,贪婪解码看起来最"稳定可靠")。但项目的实验设计是"每种方法重复跑 3 次(对应 3 个不同 seed),取 3 次的均值",目的是体现"小模型结果存在波动"这个特点。如果 Direct/CoT 用 temperature=0,3 次重复会因为贪婪解码完全一致,"重复 3 次取均值"这个设计对它们就是摆设,没有统计意义。
- **最终决定**:Direct/CoT 也改用非零 temperature(0.6,比 Self-Consistency 的 0.7 略低,因为不需要靠随机性覆盖更多候选答案再投票,只是需要"有波动"),让 3 次重复真的产生差异,同时也符合任务文档"小模型结果有波动,建议多次运行"的要求。

### 3. seed 是什么:控制"随机性是否可复现"的起始值

计算机里没有真正的随机数,所谓"随机数生成器"本质是一套**确定性的数学公式**,每次用上一个数算出下一个数。举一个极简化的例子(线性同余法):

```
下一个数 = (当前数 × 5 + 3) mod 16
```

如果起始值(seed)= 1:代入公式,第一个数是 `(1×5+3) mod 16 = 8`,用 8 再算下一个是 `(8×5+3) mod 16 = 11`,依次推下去,能生成一整条固定数列 `8, 11, 10, ...`。

如果 seed = 2:第一个数变成 `(2×5+3) mod 16 = 13`,后面整条数列都跟着完全不同:`13, 4, ...`。

公式一模一样,只是起点(seed)不同,结果就是完全不同的两条数列——而且这种"下一个数由上一个数推出来"的递推关系,对起点的变化极度敏感,起点差 1,没几步数列就完全对不上了。

### 4. seed 怎么具体控制模型生成的内容

回到第 1 点:模型每选一个词,都要从"候选词概率表"里抽签选一个。这个"抽签"动作具体怎么做——程序会从 seed 决定的那条数列里,按顺序取下一个数,把它映射成 0~1 之间的一个值,再看这个值落在概率表的哪个区间,就选中对应的词。

所以"seed 决定了一整条抽签用的数字序列"这句话,落到实际效果就是:**同一个 seed,从生成第一个词到最后一个词,每一步抽到的数字都完全一样,最终整段生成的文字也会一字不差地复现**;换一个 seed,每一步抽到的数字全变,即使 temperature 完全一样,生成的内容也会走上完全不同的路。

这也解释了一件听起来矛盾、实际不矛盾的事:模型输出"看起来很随机"(不知道 seed 的人完全看不出规律,猜不到下一步会生成什么),但只要固定 seed,它又是"确定可复现"的(同一 seed 重跑,结果分毫不差)。这两者不矛盾,因为本质上走的是同一套数学计算,只是"知道起点"和"不知道起点"的人看到的结果不一样——知道起点的人(掌握 seed 的人)能精确复现,不知道起点的人只能看到一串看不出规律的输出。

### 5. seed 在这个项目里怎么配的

`src/config.py` 里:

```python
DEFAULT_SEEDS: list[int] = [41, 42, 43]
```

每种方法(Direct/CoT/Self-Consistency)都重复跑 3 次,分别对应这三个 seed。这么设计要同时满足两个目标:

- **体现真实的采样波动**:配合上面第 2 点的非零 temperature,3 次重复因为 seed 不同,真的会抽到不同的数字序列、走上不同的生成路径,得到有波动的 3 组结果,这些波动才是任务要求"小模型结果存在波动,建议多次运行"背后想观察的东西。
- **保证实验完全可复现**:虽然 3 次重复之间有波动,但只要 seed 列表固定成 `[41, 42, 43]`,任何人拿着同样的代码重跑,每一次(seed=41 对 seed=41,seed=42 对 seed=42……)都应该得到完全一样的结果。这对写实验报告、答辩被追问"这个数字怎么来的"时能不能现场复现,很重要——"有波动"和"可复现"看起来像两个相反的要求,靠"seed 固定但取多个不同的 seed"就能同时满足。

## 2026-09-18:思考模式(enable_thinking)在三种方法里为什么这么配

### 1. enable_thinking 是什么:控制模型是否进入"思考阶段"的开关

项目用的 Qwen3.5-9B([[project-config]] 见 `src/config.py` 的 `model_path`)是一种"混合思考"模型:正式回答之前,它可以先输出一段包在 `<think>...</think>` 标签里的内部推理过程(思考阶段),模型会参考这段内容再组织最终答案,但这段内容通常不算"正式给用户看的回答"。`chat_template.jinja` 通过 `enable_thinking` 这个参数决定要不要留出这个阶段,`ModelRunner.build_prompt`(`src/model_runner.py:99`)把它转成两种行为:

- **不传(None,默认)**:模板走"参数未定义"的分支,插入一段**没有闭合**的 `<think>\n`。相当于把"要不要思考、思考多久"完全交给模型自己判断——模型可以自己选择先写一大段思考再给答案,也可以很快结束思考直接回答。
- **显式传 False**:模板插入一段**已经闭合好**的空 `<think>\n\n</think>\n\n`。模型拿到的 prompt 里,思考阶段从一开始就已经"结束"了,它看到的下一步只能是直接给正式回答——相当于在 prompt 层面把思考阶段直接堵死,强制跳过。

### 2. 三种方法具体怎么配的,为什么(`src/methods/base.py:56`、`direct.py:7`、`cot.py:7`、`self_consistency.py:38`)

- **Direct:显式传 `enable_thinking=False`**。Direct 方法在实验设计里的定位是"不给推理空间,直接输出答案"的下限对照组,三种方法的对比本质上就是想看"有没有推理过程"这一个变量带来的差异。但 Qwen3.5 是混合思考模型,如果不显式关闭,即使 prompt 只要求直接给答案,模型也可能自己选择先思考一段再回答——这样 Direct 实际上也发生了内部推理,和 CoT 在"是否推理"这个维度上就没区别了,三种方法的对比就失去了意义。所以必须显式传 `False`,从 prompt 层面强制堵死思考阶段,确保 Direct 是真正意义上"不含推理过程"的基线。
- **CoT / Self-Consistency:都不传这个参数,保持默认 `None`**。这两种方法的设计目的就是考察"有推理过程"时的效果(CoT 显式要求模型逐步推理并写出来;Self-Consistency 建立在 CoT 的 prompt 之上,多次采样再投票),所以没必要也不应该关掉模型自带的思考能力——让模型自己决定要不要用、用多长,比强行开或强行关都更贴近真实使用场景,也符合"CoT/SC 就是要看推理"这个实验意图。

一句话总结这组设计:**enable_thinking 是"要不要给模型留推理空间"的开关,三种方法里只有 Direct 需要主动关掉它,CoT 和 Self-Consistency 都靠模型自主决定,因为它们本来就是在考察推理这件事本身。**

## 2026-09-19:vLLM 跑起来时的两条进度条,"Rendering prompts" 和 "Processed prompts" 分别在干什么

跑实验时终端会先后出现两条 tqdm 进度条,对应 vLLM 源码 `vllm/entrypoints/llm.py` 里两个不同阶段(装的版本是 0.19.1,行号可能随版本变化):

1. **Rendering prompts**(`_add_completion_requests`):把每条 prompt 字符串分词成 token id,塞进 engine 的请求队列。纯 CPU 操作,逐条处理,耗时基本只跟 prompt 数量、长度成正比,一般很快跑完。

   注意这里容易搞混的一点:项目里 `ModelRunner.build_prompt()`(`src/model_runner.py:100`)已经用 tokenizer 的 `apply_chat_template` 把 messages 渲染成了字符串,这一步做的是 messages → prompt 字符串。vLLM 这条 "Rendering prompts" 进度条是对这个字符串**再做一次分词**(prompt 字符串 → token id),两者不是同一步,只是名字里都有"render"容易联想到一起。

2. **Processed prompts**(`_run_engine`):真正的生成主循环,`while llm_engine.has_unfinished_requests(): engine.step()`,每一步做一次 prefill/decode,在 GPU 上跑。这条进度条的总耗时 ≈ 这一批所有请求的(输入 token + 输出 token)之和,几乎就是整次推理调用的耗时主体。

**为什么"有时候很长"**:基本都是 Processed prompts 这一条在拖时间,不是 Rendering。Self-Consistency 每题要采样 `n=5` 份,总生成量约是 CoT 的 5 倍(实测 avg_output_tokens 一个 8000+ 一个 40000+,比例接近 5),所以同样题量下 Self-Consistency 跑得明显更久;另外 Level 5 难题开思考模式后单条生成本身就长,`max_tokens` 调得越大,理论上单条生成能跑的时间也越长,Processed prompts 这条进度条走得就越慢。

### 为什么两条都叫 "...prompts"

两条进度条数的是同一批"请求"(对应你传进去的每一条 prompt),只是站在不同阶段数:Rendering 数的是"这条 prompt 有没有被分好词、注册成请求";Processed 数的是"这条请求有没有跑完生成"。同一个单位(prompt/请求),前一个阶段处理它的输入,后一个阶段等它的输出完成。

### `output_tokens == max_tokens` 的实锤:q161

`cot/run_seed42/per_question.jsonl` 里 q161 这条:`"prediction": null, "output_tokens": 20000`——当时 `max_tokens` 恰好设的是 20000。这条生成不是模型自己说完话停下来的(那样 `finish_reason` 会是 `"stop"`,数字不会刚好等于上限),是被 `while engine.has_unfinished_requests(): engine.step()` 这个循环硬控制在第 20000 步标记"完成"的(`finish_reason == "length"`)。这一条直接证明:一条请求什么时候能被 Processed prompts 这条进度条计入"完成",是受 `max_tokens` 直接约束的——模型自己停,或者撞上限,谁先到就按谁算完成。

### 每一步 `engine.step()` 到底在干什么:模型给概率分布,采样器/`seed` 决定选哪个词

`step()` 每喊一次,只往每条未完成的请求后面添一个 token,不是一次吐出一整段话。具体到这一步内部:

1. **模型只负责算一件事**:给定当前已经生成的所有内容,下一个位置每个候选词各有多大概率——模型的输出本质是一张"候选词 → 概率"的表,不是直接给出"就是这个词"的结论。
2. **真正"选哪个词"是采样器(sampler)的工作**,不是模型本身决定的。采样器按 `temperature`(把这张概率表变得更平滑/更尖锐)和 `seed`(决定"抽签"用的那条确定性数字序列,详见上面 seed 那节 [[project-seed-note]])算出一个具体的词,追加到序列末尾。
3. 这个新词被喂回模型,模型基于"原内容 + 新词"重新算下一步的概率表,`step()` 再喊一次——如此循环,直到某条请求生成了停止符(`finish_reason: "stop"`)或撞到 `max_tokens`(`finish_reason: "length"`),这条请求才算完成,从 Processed prompts 的"未完成集合"里退出。

所以完整链路是:**模型每一步只产出概率分布 → 采样器(受 temperature/seed 控制)从分布里选出具体的词 → 循环喂回、逐词累积 → 直到停止符或 `max_tokens` 兜底,请求才算"处理完"**。Processed prompts 这条进度条,数的就是这个循环里"退出未完成集合"的请求数量。

### 没开思考模式、单条 output_tokens 也不算多,为什么 Processed prompts 还是走了很久

这是两件事叠在一起,跟"有没有思考模式"本身没关系:

1. **decode 天生是逐 token 串行的,跟有没有 `<think>` 标签无关**。不开思考模式,模型正常写解题过程(展开算式、分情况讨论)照样要写几千字,每个 token 都要过一次完整的模型前向计算——吐 3000 个 token 就要老老实实跑 3000 步 `engine.step()`,这个耗时只取决于"要写多少字",不取决于"是不是思考模式产生的"。

2. **`engine.step()` 是让这一批所有未完成的请求一起往前挪一步,不是谁写完谁先退出**。Self-Consistency 的 `n=5` 意味着一道题就有 5 条序列同时在跑,sanity check 一批 5 道题就是最多 25 条序列同时占着显存、同时被这个循环喂 token。只要这 25 条里还有一条没写完(比如卡到近 5000 token 才停),整个 batch 就要陪着它一步步跑到底——Processed prompts 不会因为其他几条已经写完就提前挪动,它卡的是"这一批里最慢的那条"。

所以感觉"token 不多但等了很久",往往是:单条看起来不夸张,但一批里有几条偏长的拖了后腿,而且是多条序列一起占着 GPU 做前向计算,不是纯粹被某一条的 token 总数决定的。

### 把 max_tokens 调到 32000 后,truncation 并没有清零:q161 换了个方式继续截断

一开始以为把 `max_tokens` 从 20000 调到 32000 就解决了截断问题,但 `20260919_184838_` 这一轮 `direct`(truncation_rate 0.2)和 `self_consistency`(truncation_rate 0.12)里都还有截断,只有 `cot` 这一轮碰巧是 0。查下来两边的截断都来自同一道题:**q161(level 5)**。

- `direct/run_seed42/per_question.jsonl` 里 q161:`output_tokens: 32000`,`finish_reason: ["length"]`——精确等于当前的 `max_tokens` 上限,和当年 20000 那次一模一样的信号:不是模型自己停的,是被硬截断的。
- `self_consistency/run_seed42/per_question.jsonl` 里 q161 的 5 个采样,`finish_reasons` 是 `["length","stop","length","length","stop"]`,`samples`(每条的抓取结果)是 `[null, "14", null, null, "14"]`——5 条里 3 条截断、2 条正常收敛到 `\boxed{14}`,靠 vote 兜住了正确答案。

关键是这次去翻了被截断的 3 条原文,**不是** 之前那种"答案写完了还在死循环"的退化:每条 1500 多行文本里,重复 3 次以上的行只有十几条、最多重复十次左右,绝大多数行都是新内容,是模型在穷举各种根的组合($x=-33,-11,3,...$ 各种取值分情况讨论)。也就是 `config.py` 注释里说的那种情况——**真实的分支探索,不是重复循环**,只是这道题分支确实多,32000 个 token 写到一半还没探索完就被截断,根本没来得及写 `\boxed{}`。

所以结论是:`repetition_penalty=1.05` 是在压"同一句话反复讲"这种退化循环,对 q161 这种"分支本身就多、写不完"的情况没用武之地——不管 `max_tokens` 调多大,这道题都有可能刚好不够用。这类 case 目前的处理方式是让它保持 `truncation_rate` 不为 0,靠 `extract_answer.py` 的 `allow_fallback`(按 `finish_reason == "length"` 关掉兜底)诚实返回 `None`,而不是从截断处硬凑一个假答案;`self_consistency` 还能靠多个样本投票把这类偶发截断吃掉。真要把 q161 的 truncation_rate 压到 0,得从"限制分支探索的写法"或者"接受这道题在 direct/cot 下就是会失败"这个方向想,单纯加大 `max_tokens` 治不了。

---

## 2026-09-22: 三个基线方法的实验结果对比与启示

### 实验背景

在 Qwen3.5-9B(tensor_parallel_size=2)上,对 MATH-500 的 200 题子集,跑了三个基线方法:**Direct**(直接回答)、**CoT**(逐步推理)、**Self-Consistency**(N=5投票)。每个方法用 3 个 seed(41/42/43)各跑一遍,共 9 次实验完成。

### 核心数据

| 指标 | Direct | CoT | Self-Consistency |
|------|--------|-----|------------------|
| 平均准确率 | 94.17% | 94.00% | **95.00%** |
| 准确率稳定性 | ±0.76% | ±0.00% | ±0.00% |
| 平均 token/题 | 2,602 | 7,209 | 36,143 |
| Token 效率* | **0.362** | 0.130 | 0.026 |
| 截断率 | 4.33% | 2.50% | 2.00% |

*Token 效率 = 准确率 / (平均 tokens ÷ 1000)

### 五个关键发现

#### 1. **Self-Consistency 投票机制的稳定性压倒一切**

Self-Consistency 的 3 个 seed 精确相同都是 95.00% 准确率(**变异系数 0%**),而 Direct 在三个 seed 之间波动 ±0.76%(93.50% 到 95.00%)。

**为什么**:Self-Consistency 通过 N=5 投票把单个样本的波动平滑掉了。CoT 之所以也是 ±0.00% 是因为三个 seed 的准确率碰巧都是 94.00%,不是说 CoT 本身更稳定——从数据上看 Direct 在三个 seed 实际表现上更有波动,但投票机制吃掉了这种波动。

**启示**:虽然 Self-Consistency 的平均 token 成本最高(36k),但用投票换稳定性在某些场景下值得。对后续的 Debate/PEV 设计来说,多智能体协作如果能像投票一样"消除个别样本的歪斜",可能也会有类似的稳定性收益。

#### 2. **Direct 是 token 效率之王,但代价是准确率偏低**

Direct 每题平均 2,602 token,是 CoT 的 36%、Self-Consistency 的 7.2%。但它的准确率(94.17%)介于 CoT(94.00%)和 Self-Consistency(95.00%)之间。

**Token 效率对比**:
- Direct: **0.362** (最高效)
- CoT: 0.130
- Self-Consistency: 0.026 (最低效)

**启示**:在资源受限的场景(比如线上推理延迟/成本敏感),Direct 虽然给不了最高准确率,但性价比最优。对多智能体方法来说,如果多个简单智能体(比如 Debate 里的三个 Solver)不投更多 token,纯靠"协作机制"来提升准确率,那这部分的 token 成本不会显著增加——价值就在于协作的 overhead 能不能通过改进答案质量收回来。

#### 3. **CoT 的隐形优势:截断率最低**

CoT 的截断率是 2.50%(最低),Direct 是 4.33%(最高),Self-Consistency 是 2.00%(次低)。

这看起来和"CoT 的 token 数最多"矛盾——token 多应该更容易截断。但实际上:

| 方法 | 平均 output_tokens | 截断率 | 推论 |
|------|------------------|--------|------|
| Direct | 2,568 | 4.33% | 虽然 token 少,但某些难题一次就探索完,会碰到"该截断却没截"的罕见情况? |
| CoT | 7,261 | 2.50% | token 更多反而截断少,说明 CoT 的输出模式更"温和",不容易一次性冲满 32k token |
| Self-Consistency | 35,407 | 2.00% | N=5 投票制也有同样效果 |

**启示**:截断率的高低不完全由平均 token 决定,和**输出的方差**也有关。Direct 虽然平均 token 少,但偏差大(有的题目很短,有的题目长),所以在 seed 42 这次碰巧有题目接近 32k 上限就被截断;CoT 更"机械"地按步骤推理,输出更均匀,反而不容易撞上上限。

对 Debate/PEV 的设计含义:**多轮对话** 能不能通过"每轮都短一点,分散成多轮"的方式,既保证充分讨论,又不容易触发单轮截断?

#### 4. **Seed 41/42/43 的微观波动反映出采样的本质随机性**

Direct 三个 seed 的准确率分别是 94.00% / 95.00% / 93.50%,虽然平均只差 1.5%,但这 1.5% 在 200 题里就是 3 题的差异(题目被模型判错/判对的概率在不同采样下确实会变化)。

CoT 和 Self-Consistency 各 seed 都相同,看起来像是巧合(**种子本身没有导致输出差异**),但更可能是:
- **CoT**: 虽然也是采样,但逻辑推理的链条更受输入/prompt 制约,seed 的影响通过 temperature 放大的空间有限
- **Self-Consistency**: 5 条投票把单条样本的波动吃掉了

**启示**:多智能体协作里,如果各个智能体都用同一个 seed,或者多个 seed 之间的差异被聚合消除(如投票、取中位数),那结果也会表现出这种"表面稳定"。但这不等于系统本身稳定——只是把波动隐藏在内部了。

#### 5. **准确率天花板可能就在 95% 左右**

三个方法中最高的是 Self-Consistency 的 95.00%。Direct 和 CoT 都在 94% 附近。

这可能反映出:
- **200 题子集本身的难度分布** 可能有 5% 左右的题目是 9B 模型在这个 temperature/max_tokens 设置下"天然答不对"的(可能需要更大模型、或者更细致的 prompt 设计)
- **Self-Consistency 的投票**能从平均 94% 提升到 95%,gain 只有 1%,说明 N=5 能吃掉的波动有限

**启示**:后续跑 Debate/PEV 时,如果准确率能突破 95% 进入 96%+,那说明多智能体协作真的在"互相纠正"上有效果(不是单纯的投票放大);如果卡在 95% 以下,要考虑是协作机制有问题,还是这道题对 9B 模型本身就太难。

### 一句话总结

**单样本有波动,投票消波动,但代价是 token 成本翻倍。后续多智能体方法的价值衡量标准:能否用比投票更省 token 的协作机制来逼近或超过 Self-Consistency 的 95%+准确率。**

---

## 2026-09-22: 排查 thinking 模式有没有生效,搜错了标签方向

### 现象

跑完三个基线后回头想确认:CoT/Self-Consistency 到底有没有用上 Qwen3.5-9B 的原生 thinking 模式(`enable_thinking=None` 是交给模型自己决定)。第一次排查直接 `grep "<think>" per_question.jsonl`,结果 6 个 sanity_check 目录 + 正式跑的结果全都是 **0 命中**,于是先得出了一个错误结论:"模型没用 thinking,重不重跑无所谓"。

### 搞错的地方

`<think>` **开标签不会出现在模型的输出里**。看 `src/model_runner.py:100-108` 的 `build_prompt`:

```python
def build_prompt(self, messages: list[dict], enable_thinking: bool | None = None) -> str:
    """enable_thinking=None(默认)时不传这个参数给模板,由模型自己决定要不要思考
    (对应 chat_template.jinja 里 `enable_thinking is defined` 为假的分支,插入未闭合
    的 `<think>\n`)。显式传 False 会让模板插入已闭合的空 `<think>\n\n</think>\n\n`,
    跳过思考阶段直接进入正式回答。"""
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
```

`enable_thinking=None` 时,chat template 会在**发给模型的 prompt 末尾**插入一个未闭合的 `<think>\n`,相当于"提示模型:接下来你可以开始思考"。这个开标签是 prompt 的一部分,不是 completion 的一部分,所以 vLLM 返回的 `raw_outputs`(也就是 `per_question.jsonl` 里存的内容)**天然不会包含这个开标签**——不管模型有没有真的思考,搜开标签永远是 0,这个信号本身就没有区分力。

真正该搜的是模型**自己写的闭合标签** `</think>`:如果模型选择走 thinking 分支,它会在思考完之后自己吐出 `</think>` 再进入正式回答;如果 `enable_thinking=False`(模板已经塞了闭合的空 `<think>\n\n</think>\n\n`),模型不会再自己写一次。

### 重新验证后的真实结论

```bash
grep -c "</think>" results/cot/run_seed41/per_question.jsonl              # 197/200
grep -c "</think>" results/self_consistency/run_seed41/per_question.jsonl # 200/200
grep -c "</think>" results/direct/run_seed41/per_question.jsonl           # 3/200(强制关闭后的噪声)
```

**CoT 和 Self-Consistency 几乎每一题都触发了 thinking 模式**,只有 `direct` 是真正被强制关闭的。也就是说之前"CoT vs Direct"的对比,实际上是"prompt 层面的 CoT 要求 + 模型原生 thinking"叠加效果 vs "无 thinking 的直接回答",没有把这两个变量隔离开。

### 教训

1. **判断某个 tag 有没有生效,先搞清楚这个 tag 是加在 prompt 侧还是 completion 侧**,搜错了地方会拿到一个"看似确定、实际毫无信息量"的全零结果,还很容易顺着这个假结果编出一个错误但自洽的解释。
2. **第一次分析错了要主动纠正,不要因为"看起来已经有个说得通的结论"就停下来**——这次是用户追问"你确定吗,sanity check 时我明明看到过 thinking" 才发现问题,说明"数据摆在眼前"不代表"分析角度是对的"。
3. 处理办法:不重跑已有数据(占的 GPU 时间已经花了,结果依然有效,只是要标注清楚"CoT/Self-Consistency 默认允许 thinking"),额外补跑一组 `--disable-thinking` 的对照数据存到 `results_no_thinking/`(用 `--results-dir` 隔离,不覆盖原结果),两组数据都留着,具体命令见 `EXPERIMENT_COMMANDS.md`。

---

## 2026-09-23: PEV 第一次 sanity check 崩溃——策略解析没校验数量,把 thinking 草稿也切成了"策略"

### 现象

`debate`/`pev` 接入 `METHOD_RUNNERS` 后第一次真实上 GPU 跑 sanity check:

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_experiment.py --method pev --limit 5 --seeds 41 --results-dir results/sanity_check
```

前两批 generate 调用都正常跑完(第一批 5 个 prompt,应为 planner;第二批 **104 个** prompt),第三批(verifier)刚起步就崩:

```
vllm.exceptions.VLLMValidationError: This model's maximum context length is 36864 tokens.
However, you requested 0 output tokens and your prompt contains at least 36865 input tokens,
for a total of at least 36865 tokens.
```

第二批 104 个 prompt 是第一个异常信号:`--limit 5` 题、`PEVConfig.num_strategies=2`,executor 阶段正常应该是 5×2=10 个 prompt,104 个明显不对。

### 根因(委托 Codex 定位,只读排查未改代码)

1. **`src/multi_agent/roles.py` 的 `_parse_strategies(text, expected)` 完全没用 `expected` 参数做数量校验**,而是对全文任意位置出现的 `Strategy N:` 都切一段,包括模型 thinking 草稿里提到的字样、以及第一个标记之前的所有前导文本也被当成一段策略。Codex 用最小复现验证:一段"thinking 草稿提到 Strategy 1/2 + `</think>` 后正式给出 Strategy 1/2 最终版"的文本,被解析成了 **5 个**"策略"(而不是预期的 2 个)。
2. executor 阶段对解析出的每个片段都发一次请求,没有任何数量上限,5 题 × (每题被解析出的策略数,含垃圾片段)累加起来正好是 104。
3. verifier 阶段的 prompt 会把"该题全部候选策略的原文 + executor 完整输出"原样拼接进去(`roles.py:304-325`)——候选数量本该是 2,失控后变成 5+,加上每份都是未剥离 thinking 的完整原文,单题 prompt 直接冲上 36864 上限。
4. 报错信息里 `0 output tokens`/`36865` 不是项目代码把生成预算 clamp 到了 0——那是 vLLM 内部输入校验用的独立 `TokenizeParams.max_output_tokens`(默认值就是 0),被拒绝时打印的是这个字段,不代表项目自己算出的 `max_tokens`(固定 32000,详见 `config.py`/`roles.py:22`)。`36865` 只是"探测到超限即报"的下界,真实长度可能远超此数。真正的问题是:`src/model_runner.py` 的 `generate()` 从头到尾没有对 prompt 长度做任何计算/预算管理,超没超都直接发请求。
5. 连带发现一个独立 bug:`scripts/run_experiment.py` 的 `--disable-thinking` 只传给了三个基线方法,**没有传给 `debate`/`pev`**(`MULTI_AGENT_METHODS` 分支里 `run_fn(runner, questions, seed=seed)` 没带这个参数)。也就是说这次 sanity check 命令里就算加了 `--disable-thinking` 对 PEV 也不会生效,thinking 依然开着——这不是这次崩溃的根因(根因是策略解析没做数量校验),但会放大问题(thinking 草稿越长,被误切进"策略"列表的垃圾片段就越多)。

### 教训

1. **多智能体角色间的输出解析,必须用传入的期望值(`expected`)做强校验,不能只靠正则切分"看起来像"的模式**——模型在 thinking 模式下会在草稿里提前"预告"格式关键字(比如提到 `Strategy 1`/`Strategy 2` 打草稿),纯文本模式匹配分不清"这是草稿里提到的" vs "这是正式给出的最终策略"。
2. **多智能体协作的 prompt 拼接(把上一阶段完整输出喂给下一阶段)必须有长度预算意识**,不能假设"角色输出不会太长"——尤其是没有关闭 thinking 时,单次输出可能就有几千 token,多个角色/多个候选原样拼接极易线性甚至更快地吃满 context window。
3. **观察 batch 数量是最便宜的健康检查**:跑 sanity check 时,`--limit 5` 配 `num_strategies=2` 应该让人一眼看出"第二批该是 10 个 prompt",104 这个数字在日志刷过去的瞬间就该被抓出来,不用等到第三批真的报错才发现问题。
4. **给多智能体分支加新参数(比如 `--disable-thinking`)时,要检查所有分支入口是不是都接上了**,不能只改其中一条调用路径——这次 `run_experiment.py` 里 `MULTI_AGENT_METHODS` 和普通方法用的是两条不同的调用分支,加参数时漏了一条。
5. 修复顺序(还没做,下一步):先修 `_parse_strategies` 使其严格按 `expected` 数量提取最终答案(排除 thinking 内容),再给角色间传递的上下文加预算管理(摘要/截断而非全文拼接),最后把 `--disable-thinking` 接到 `debate`/`pev` 分支。改完后要重新跑一次同样的 `--limit 5` sanity check 验证 batch 数量和是否还会超限,再上 200 题正式跑。

---
