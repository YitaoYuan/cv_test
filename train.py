import os
import time
import copy
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from torch.futures import Future
import numpy as np
from datasets import load_dataset, Image as HfImage
# 这里新增
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from torchvision import transforms

# ----------------------------
# 1. 初始化分布式环境
# （保持不变）
# ----------------------------
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK       = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

dist.init_process_group(backend="nccl", init_method="env://",
                        rank=RANK, world_size=WORLD_SIZE)
torch.cuda.set_device(LOCAL_RANK)
device = torch.device("cuda", LOCAL_RANK)

# ----------------------------
# 2. 定义通信钩子
# （保持不变）
# ----------------------------
class HookState:
    def __init__(self):
        self.total_comm_time = 0.0  # 累计 all‐reduce 时间
def timing_comm_hook(state: HookState, bucket) -> Future[torch.Tensor]:
    t0 = time.perf_counter()
    pg = dist.distributed_c10d._get_default_group()
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    work = pg.allreduce([bucket.buffer()], opts)
    fut = work.get_future()            # torch._C.Future
    def _callback(fut):                # 这里入参也是一个 Future
        res_list = fut.value()         # 拿到真正的 list[Tensor]
        state.total_comm_time += time.perf_counter() - t0
        return res_list[0]             # 返回第 0 个 Tensor
    return fut.then(_callback)         # 返回 Future[Tensor]

# ----------------------------
# 3. 准备数据集与 DataLoader
#    用本地 parquet（plain_text）文件
# ----------------------------
# 你的目录结构：
#   ./cifar10/
#     └─ plain_text/
#         ├─ train-00000-of-00001.parquet
#         └─ test-00000-of-00001.parquet
def load_image(raw):
    """
    把各种形式的 raw 转成 PIL.Image.Image
    """
    # 1) 已经是 PIL
    if isinstance(raw, Image.Image):
        return raw
    # 2) bytes-like
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw))
    if isinstance(raw, memoryview):
        return Image.open(io.BytesIO(raw.tobytes()))
    # 3) numpy array
    if isinstance(raw, np.ndarray):
        return Image.open(io.BytesIO(raw.tobytes()))
    # 4) list
    if isinstance(raw, list):
        if not raw:
            raise ValueError("Empty raw list cannot be decoded as image")
        elem0 = raw[0]
        # 4.1) list of PIL
        if isinstance(elem0, Image.Image):
            return elem0
        # 4.2) list of bytes-like or ints
        #      先把 raw 扁平成一维 int/bytes，再转 bytes
        flat = []
        for x in raw:
            if isinstance(x, Image.Image):
                return x  # 如果某个元素是 PIL，就直接用它
            if isinstance(x, (bytes, bytearray, memoryview)):
                flat.extend(bytearray(x))
            elif isinstance(x, int):
                flat.append(x)
            elif isinstance(x, list) or isinstance(x, tuple):
                # 二级嵌套
                for y in x:
                    if isinstance(y, int):
                        flat.append(y)
                    elif isinstance(y, (bytes, bytearray, memoryview)):
                        flat.extend(bytearray(y))
            else:
                # 跳过不能识别的类型
                pass
        return Image.open(io.BytesIO(bytes(flat)))
    # 5) 其他全抛错
    raise TypeError(f"Unsupported raw image type: {type(raw)}")

parquet_dir = "./cifar10/plain_text"
data_files = {
    "train": f"{parquet_dir}/train-00000-of-00001.parquet",
    "test":  f"{parquet_dir}/test-00000-of-00001.parquet"
}
hf_ds = load_dataset("parquet", data_files=data_files)
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914,0.4822,0.4465],
                         [0.2470,0.2435,0.2616]),
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914,0.4822,0.4465],
                         [0.2470,0.2435,0.2616]),
])
def preprocess_train(example):
    img = load_image(example["img"]).convert("RGB")
    return {
        "pixel_values": train_transforms(img),
        "labels": example["label"]
    }
def preprocess_val(example):
    img = load_image(example["img"]).convert("RGB")
    return {
        "pixel_values": val_transforms(img),
        "labels": example["label"]
    }

train_ds = hf_ds["train"] \
              .map(preprocess_train, remove_columns=["img","label"], num_proc=4) \
              .with_format("torch") 
val_ds   = hf_ds["test"] \
              .map(preprocess_val, remove_columns=["img","label"], num_proc=4) \
              .with_format("torch") 
train_sampler = DistributedSampler(train_ds)
val_sampler   = DistributedSampler(val_ds, shuffle=False)

def collate_fn(batch):
    imgs   = torch.stack([ex["pixel_values"] for ex in batch])
    labels = torch.tensor([ex["labels"]       for ex in batch],
                          dtype=torch.long)
    return imgs, labels
train_loader = DataLoader(
    train_ds,
    batch_size=128,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_ds,
    batch_size=128,
    sampler=val_sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

batch = next(iter(train_loader))
print(batch[0].shape)

# ----------------------------
# 4. 构建模型、包成 DDP 并注册钩子
# ----------------------------
num_classes = 10

model = models.vgg16(pretrained=False)

# 1) 把 avgpool 从 7×7 改成 1×1
model.avgpool = nn.AdaptiveAvgPool2d((1,1))

with torch.no_grad():
    dummy = torch.zeros(1, 3, 32, 32)
    feats = model.features(dummy)
    feats = model.avgpool(feats)
    n_flatten = feats.view(1, -1).shape[1]



# 2) 把第一层全连接从 (25088→4096) 改成 (512→4096)
model.classifier = nn.Sequential(
    nn.Flatten(),                    # 把 [B, C, H, W] → [B, C*H*W]
    nn.Linear(n_flatten, 4096),     # In_features = 上面计算出来的 n_flatten
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
)

model.cuda(LOCAL_RANK)

ddp_model = DDP(model, device_ids=[LOCAL_RANK])
hook_state = HookState()
ddp_model.register_comm_hook(hook_state, timing_comm_hook)

# ----------------------------
# 5. 损失、优化器、LR Scheduler
# ----------------------------
criterion = nn.CrossEntropyLoss().cuda(LOCAL_RANK)
optimizer = optim.SGD(ddp_model.parameters(),
                      lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ----------------------------
# 6. 训练 & 验证函数
# （不变，只是从 DataLoader 拿 (imgs, labels)）
# ----------------------------
def train_one_epoch(loader):
    ddp_model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in loader:
        inputs = inputs.cuda(LOCAL_RANK, non_blocking=True)
        labels = labels.cuda(LOCAL_RANK, non_blocking=True)
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(outputs.argmax(1) == labels)
    return running_loss/len(loader.dataset), running_corrects.double()/len(loader.dataset)

def validate(loader):
    ddp_model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.cuda(LOCAL_RANK, non_blocking=True)
            labels = labels.cuda(LOCAL_RANK, non_blocking=True)
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(outputs.argmax(1) == labels)
    return running_loss/len(loader.dataset), running_corrects.double()/len(loader.dataset)

# ----------------------------
# 7. 主训练循环 & 通信耗时统计
# （保持不变）
# ----------------------------
num_epochs = 3
best_acc   = 0.0
best_wts   = copy.deepcopy(ddp_model.state_dict())
train_start = time.perf_counter()


for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    val_sampler.set_epoch(epoch)

    tloss, tacc = train_one_epoch(train_loader)
    vloss, vacc = validate(val_loader)
    scheduler.step()

    if RANK == 0:
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train loss: {tloss:.4f}, acc: {tacc:.4f}  |  "
              f"Val   loss: {vloss:.4f}, acc: {vacc:.4f}")
        if vacc > best_acc:
            best_acc = vacc
            best_wts = copy.deepcopy(ddp_model.state_dict())

train_end = time.perf_counter()

if RANK == 0:
    total_time = train_end - train_start
    comm_time  = hook_state.total_comm_time
    print("\n=== Training complete ===")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Total all‐reduce time: {comm_time:.1f}s  "
          f"({comm_time/total_time*100:.2f}%)")
    ddp_model.load_state_dict(best_wts)
    torch.save(ddp_model.module.state_dict(), "best_vgg16_ddp.pth")

dist.destroy_process_group()
