"""
基于 JSON 元信息创建符号链接目录结构,适配 AnomalyDINO 项目

使用示例:
    python create_symlink_structure.py \
        --json_dir /path/to/json/files \
        --image_dir /path/to/images \
        --output_dir data/symlink_dataset
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import argparse


@dataclass
class MetaSample:
    image_path: str
    mask_path: str | None
    label: bool  # True 为异常, False 为正常
    anomaly_class: str  # 异常类型名称


def load_category_data(json_dir: Path, image_dir: Path) -> Dict[str, Dict[str, List[MetaSample]]]:
    """
    从 JSON 文件加载所有类别的数据
    
    返回格式:
    {
        "category1": {
            "train": [MetaSample, ...],
            "test": [MetaSample, ...]
        },
        "category2": {...}
    }
    """
    category_datas: Dict[str, Dict[str, List[MetaSample]]] = {}
    
    for json_file in json_dir.glob("*.json"):
        print(f"Loading dataset from {json_file}...")
        with open(json_file, "r") as f:
            data = json.load(f)
        
        normal_class = data["meta"]["normal_class"]
        prefix: str = data["meta"]["prefix"]
        category: str = json_file.stem
        
        train_samples: List[MetaSample] = []
        test_samples: List[MetaSample] = []
        
        # 处理训练集
        if "train" in data:
            for item in data["train"]:
                anomaly_class = item.get("anomaly_class", normal_class)
                is_anomaly = anomaly_class != normal_class
                image_path = image_dir / prefix / item["image_path"]
                
                # 训练集通常只包含正常样本
                if not is_anomaly:
                    train_samples.append(
                        MetaSample(
                            image_path=str(image_path),
                            mask_path=None,
                            label=False,
                            anomaly_class=normal_class,
                        )
                    )
        
        # 处理测试集
        if "test" in data:
            for item in data["test"]:
                anomaly_class = item["anomaly_class"]
                is_anomaly = anomaly_class != normal_class
                image_path = image_dir / prefix / item["image_path"]
                mask_path = (
                    image_dir / prefix / item["mask_path"] if is_anomaly and "mask_path" in item
                    else None
                )
                
                test_samples.append(
                    MetaSample(
                        image_path=str(image_path),
                        mask_path=str(mask_path) if mask_path is not None else None,
                        label=is_anomaly,
                        anomaly_class=anomaly_class,
                    )
                )
        
        category_datas[category] = {
            "train": train_samples,
            "test": test_samples
        }
        
        print(f"  - Category '{category}': {len(train_samples)} train, {len(test_samples)} test samples")
    
    return category_datas


def get_anomaly_type(item: dict, normal_class: str) -> str:
    """从 JSON item 中提取异常类型名称"""
    anomaly_class = item.get("anomaly_class", "unknown")
    if anomaly_class == normal_class:
        return "good"
    return anomaly_class


def create_symlink_structure(
    json_dir: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False
):
    """
    基于 JSON 元信息创建符号链接目录结构
    
    Args:
        json_dir: 包含 JSON 元信息文件的目录
        image_dir: 原始图像根目录
        output_dir: 输出符号链接目录结构的目录
        overwrite: 如果输出目录已存在是否覆盖
    """
    json_dir = Path(json_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    if output_dir.exists() and not overwrite:
        print(f"Warning: Output directory {output_dir} already exists.")
        response = input("Continue and merge? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载所有类别数据
    category_datas = load_category_data(json_dir, image_dir)
    
    # 统计信息
    total_symlinks = 0
    missing_files = 0
    
    # 为每个类别创建符号链接结构
    for category, phases in category_datas.items():
        print(f"\nProcessing category: {category}")
        
        # 处理训练集
        train_samples = phases["train"]
        train_dir = output_dir / category / "train" / "good"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        for sample in train_samples:
            img_path = Path(sample.image_path)
            if not img_path.exists():
                print(f"  Warning: Image not found: {img_path}")
                missing_files += 1
                continue
            
            link_path = train_dir / img_path.name
            if not link_path.exists():
                os.symlink(img_path.absolute(), link_path)
                total_symlinks += 1
        
        print(f"  - Created {len(train_samples)} train symlinks")
        
        # 处理测试集
        test_samples = phases["test"]
        
        # 按类别分组
        grouped_samples: Dict[str, List[MetaSample]] = {}
        for sample in test_samples:
            # 正常样本统一归为 'good'
            if not sample.label:
                anomaly_type = "good"
            else:
                anomaly_type = sample.anomaly_class
            
            if anomaly_type not in grouped_samples:
                grouped_samples[anomaly_type] = []
            grouped_samples[anomaly_type].append(sample)
        
        # 创建测试集符号链接
        for anomaly_type, samples in grouped_samples.items():
            test_dir = output_dir / category / "test" / anomaly_type
            test_dir.mkdir(parents=True, exist_ok=True)
            
            for sample in samples:
                img_path = Path(sample.image_path)
                if not img_path.exists():
                    print(f"  Warning: Image not found: {img_path}")
                    missing_files += 1
                    continue
                
                link_path = test_dir / img_path.name
                if not link_path.exists():
                    os.symlink(img_path.absolute(), link_path)
                    total_symlinks += 1
                
                # 只为异常样本创建掩码
                if sample.label and sample.mask_path:
                    mask_src = Path(sample.mask_path)
                    if mask_src.exists():
                        mask_dir = output_dir / category / "ground_truth" / anomaly_type
                        mask_dir.mkdir(parents=True, exist_ok=True)
                        
                        mask_link = mask_dir / img_path.name
                        if not mask_link.exists():
                            os.symlink(mask_src.absolute(), mask_link)
                            total_symlinks += 1
                    else:
                        print(f"  Warning: Mask not found: {mask_src}")
                        missing_files += 1
            
            print(f"  - Created {len(samples)} test symlinks for '{anomaly_type}'")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  - Categories processed: {len(category_datas)}")
    print(f"  - Total symlinks created: {total_symlinks}")
    print(f"  - Missing files: {missing_files}")
    print(f"  - Output directory: {output_dir.absolute()}")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create symlink structure for AnomalyDINO from JSON metadata"
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        required=True,
        help="Directory containing JSON metadata files (*.json)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Root directory of the original image dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for symlink structure"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory without confirmation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    create_symlink_structure(
        json_dir=args.json_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite
    )
    
    print("Done! You can now use the symlink structure with AnomalyDINO:")
    print(f"  python run_anomalydino.py --data_root {args.output_dir} --dataset YourDataset")
