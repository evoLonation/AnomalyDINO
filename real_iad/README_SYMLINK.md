# 使用符号链接适配自定义数据集到 AnomalyDINO

本指南帮助你将基于 JSON 元信息的自定义数据集适配到 AnomalyDINO 项目,无需重组原始文件结构。

## 快速开始

### 步骤 1: 创建符号链接结构

```bash
python create_symlink_structure.py \
    --json_dir /path/to/your/json/metadata \
    --image_dir /path/to/your/original/images \
    --output_dir data/your_dataset_symlinks
```

**参数说明:**
- `--json_dir`: 包含 JSON 元信息文件的目录(每个 .json 文件对应一个类别)
- `--image_dir`: 原始图像的根目录
- `--output_dir`: 输出符号链接目录的路径
- `--overwrite`: (可选) 覆盖已存在的输出目录

**输出示例:**
```
Loading dataset from /path/to/json/bottle.json...
  - Category 'bottle': 209 train, 83 test samples
Loading dataset from /path/to/json/cable.json...
  - Category 'cable': 224 train, 92 test samples
...

Processing category: bottle
  - Created 209 train symlinks
  - Created 20 test symlinks for 'good'
  - Created 21 test symlinks for 'broken_large'
  - Created 18 test symlinks for 'broken_small'
  - Created 24 test symlinks for 'contamination'
...

============================================================
Summary:
  - Categories processed: 15
  - Total symlinks created: 4952
  - Missing files: 0
  - Output directory: /path/to/data/your_dataset_symlinks
============================================================
```

### 步骤 2: 生成数据集配置代码

```bash
python register_dataset.py \
    --data_root data/your_dataset_symlinks \
    --dataset_name YourDataset
```

**输出示例:**
```
Scanning dataset structure in: data/your_dataset_symlinks
============================================================
  - bottle: 209 train, 83 test samples, 3 anomaly types
  - cable: 224 train, 92 test samples, 8 anomaly types
  ...
============================================================
Found 15 object categories
Dataset name: YourDataset
============================================================

============================================================
Add the following code to src/utils.py in get_dataset_info():
============================================================

    elif dataset == "YourDataset":
        objects = ["bottle", "cable", "capsule", ...]
        
        object_anomalies = {
            "bottle": ["broken_large", "broken_small", "contamination"],
            "cable": ["bent_wire", "cable_swap", "combined", ...],
            ...
        }
        
        # 根据预处理策略配置掩码和旋转
        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {o: False for o in objects}
        else:
            masking_default = {o: True for o in objects}
        
        if preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {o: True for o in objects}
        elif preprocess in ["informed", "masking_only", "informed_no_mask"]:
            rotation_default = {o: False for o in objects}

============================================================

Configuration also saved to: dataset_config_YourDataset.txt
```

### 步骤 3: 添加配置到 `src/utils.py`

将生成的代码复制到 `src/utils.py` 的 `get_dataset_info()` 函数中,插入位置在最后的 `else` 子句之前:

```python
def get_dataset_info(dataset, preprocess):
    # ...existing code for MVTec, VisA...
    
    elif dataset == "YourDataset":
        # 粘贴生成的配置代码
        objects = [...]
        object_anomalies = {...}
        masking_default = {...}
        rotation_default = {...}
    
    else:
        raise ValueError(f"Dataset '{dataset}' not yet covered!")
    
    # ...remaining code...
```

### 步骤 4: 运行异常检测

```bash
python run_anomalydino.py \
    --dataset YourDataset \
    --data_root data/your_dataset_symlinks \
    --model_name dinov2_vits14 \
    --resolution 448 \
    --shots 1 2 4 \
    --preprocess agnostic \
    --faiss_on_cpu
```

## JSON 文件格式要求

每个 JSON 文件应包含以下结构:

```json
{
  "meta": {
    "normal_class": "good",
    "prefix": "bottle"
  },
  "train": [
    {
      "image_path": "train/good/001.png",
      "anomaly_class": "good"
    },
    ...
  ],
  "test": [
    {
      "image_path": "test/broken_large/000.png",
      "anomaly_class": "broken_large",
      "mask_path": "ground_truth/broken_large/000_mask.png"
    },
    {
      "image_path": "test/good/010.png",
      "anomaly_class": "good"
    },
    ...
  ]
}
```

**字段说明:**
- `meta.normal_class`: 正常样本的类别名称
- `meta.prefix`: 图像路径的前缀(相对于 `--image_dir`)
- `train`: 训练集样本列表
- `test`: 测试集样本列表
- `image_path`: 图像相对路径
- `anomaly_class`: 样本的异常类别
- `mask_path`: (可选) 异常分割掩码路径

## 高级选项

### 自定义掩码和旋转策略

如果需要针对特定类别手动配置掩码和旋转:

```python
elif dataset == "YourDataset":
    objects = ["obj1", "obj2", "obj3"]
    
    # 手动配置哪些类别需要掩码
    masking_default = {
        "obj1": True,   # 使用掩码
        "obj2": False,  # 不使用掩码
        "obj3": True
    }
    
    # 手动配置哪些类别需要旋转增强
    rotation_default = {
        "obj1": False,  # 不旋转(如瓶子有固定方向)
        "obj2": True,   # 旋转(如螺丝可任意方向)
        "obj3": False
    }
```

### 只测试部分类别

```bash
# 修改 src/utils.py,只包含需要的类别
objects = ["bottle", "cable"]  # 只测试这两个类别
```

或者在代码中添加过滤:

```python
# 在 run_anomalydino.py 中
objects = [obj for obj in objects if obj in ["bottle", "cable"]]
```

## 常见问题

### Q: 符号链接创建失败?
A: 检查:
1. 原始图像路径是否正确
2. JSON 中的路径与实际文件是否匹配
3. 是否有文件权限问题

### Q: 异常类型识别错误?
A: 脚本会从图像路径的父目录名提取异常类型,确保路径格式为:
```
.../anomaly_type/image.jpg
```

### Q: 如何使用 GPU 加速 FAISS?
A: 去掉 `--faiss_on_cpu` 参数,并确保安装了 `faiss-gpu`:
```bash
conda install -c pytorch faiss-gpu
```

### Q: 内存不足?
A: 尝试:
1. 降低分辨率: `--resolution 224`
2. 使用更小的模型: `--model_name dinov2_vits14`
3. 使用 CPU 版 FAISS: `--faiss_on_cpu`

## 文件说明

- `create_symlink_structure.py`: 创建符号链接目录结构
- `register_dataset.py`: 扫描数据集并生成配置代码
- `dataset_config_*.txt`: 生成的配置代码(需手动添加到 `src/utils.py`)

## 完整工作流示例

```bash
# 1. 创建符号链接
python create_symlink_structure.py \
    --json_dir /data/meta \
    --image_dir /data/images \
    --output_dir data/my_dataset

# 2. 生成配置
python register_dataset.py \
    --data_root data/my_dataset \
    --dataset_name MyDataset

# 3. 编辑 src/utils.py,添加配置

# 4. 运行检测
python run_anomalydino.py \
    --dataset MyDataset \
    --data_root data/my_dataset \
    --shots -1 \
    --preprocess agnostic
```

## 验证数据集结构

运行前可以验证目录结构:

```bash
tree data/your_dataset_symlinks -L 3
```

期望输出:
```
data/your_dataset_symlinks/
├── bottle/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   ├── broken_large/
│   │   ├── broken_small/
│   │   └── contamination/
│   └── ground_truth/
│       ├── broken_large/
│       ├── broken_small/
│       └── contamination/
├── cable/
│   └── ...
...
```
