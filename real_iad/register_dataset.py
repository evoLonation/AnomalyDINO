"""
自动从符号链接目录结构中提取数据集信息并生成配置代码

使用示例:
    python register_dataset.py --data_root data/your_dataset_symlinks --dataset_name YourDataset
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple


def scan_dataset_structure(data_root: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    扫描数据集目录结构,自动提取对象类别和异常类型
    
    Returns:
        objects: 对象类别列表
        object_anomalies: 每个对象的异常类型字典
    """
    objects = []
    object_anomalies = {}
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    # 遍历所有对象类别
    for obj_dir in sorted(data_root.iterdir()):
        if not obj_dir.is_dir():
            continue
        
        object_name = obj_dir.name
        objects.append(object_name)
        
        # 查找测试集中的异常类型
        test_dir = obj_dir / "test"
        anomaly_types = []
        
        if test_dir.exists():
            for anomaly_dir in sorted(test_dir.iterdir()):
                if anomaly_dir.is_dir() and anomaly_dir.name != "good":
                    anomaly_types.append(anomaly_dir.name)
        
        object_anomalies[object_name] = anomaly_types
        
        # 统计样本数量
        train_count = len(list((obj_dir / "train" / "good").glob("*"))) if (obj_dir / "train" / "good").exists() else 0
        test_count = len(list(test_dir.rglob("*"))) if test_dir.exists() else 0
        
        print(f"  - {object_name}: {train_count} train, {test_count} test samples, {len(anomaly_types)} anomaly types")
    
    return objects, object_anomalies


def generate_config_code(dataset_name: str, objects: List[str], object_anomalies: Dict[str, List[str]]) -> str:
    """生成数据集配置代码"""
    
    # 生成对象列表
    objects_str = ', '.join([f'"{obj}"' for obj in objects])
    
    # 生成异常字典
    anomalies_lines = []
    for obj, anomalies in object_anomalies.items():
        anomalies_str = ', '.join([f'"{a}"' for a in anomalies])
        anomalies_lines.append(f'            "{obj}": [{anomalies_str}]')
    anomalies_dict_str = ',\n'.join(anomalies_lines)
    
    code = f'''
    elif dataset == "{dataset_name}":
        objects = [{objects_str}]
        
        object_anomalies = {{
{anomalies_dict_str}
        }}
        
        # 根据预处理策略配置掩码和旋转
        if preprocess in ["informed_no_mask", "agnostic_no_mask"]:
            masking_default = {{o: False for o in objects}}
        else:
            # 默认启用掩码,可根据实际情况手动调整
            masking_default = {{o: True for o in objects}}
        
        if preprocess in ["agnostic", "agnostic_no_mask"]:
            rotation_default = {{o: True for o in objects}}
        elif preprocess in ["informed", "masking_only", "informed_no_mask"]:
            rotation_default = {{o: False for o in objects}}
'''
    
    return code


def register_dataset(data_root: str, dataset_name: str):
    """
    扫描数据集并生成注册代码
    
    Args:
        data_root: 符号链接目录结构的根路径
        dataset_name: 数据集名称(用于 --dataset 参数)
    """
    data_root = Path(data_root)
    
    print(f"Scanning dataset structure in: {data_root}")
    print("="*60)
    
    objects, object_anomalies = scan_dataset_structure(data_root)
    
    print("="*60)
    print(f"Found {len(objects)} object categories")
    print(f"Dataset name: {dataset_name}")
    print("="*60)
    
    # 生成配置代码
    config_code = generate_config_code(dataset_name, objects, object_anomalies)
    
    print("\n" + "="*60)
    print("Add the following code to src/utils.py in get_dataset_info():")
    print("="*60)
    print(config_code)
    print("="*60)
    
    # 保存到文件
    output_file = f"dataset_config_{dataset_name}.txt"
    with open(output_file, "w") as f:
        f.write("# Add this code to src/utils.py in the get_dataset_info() function\n")
        f.write("# Insert it before the final 'else' clause that raises ValueError\n\n")
        f.write(config_code)
    
    print(f"\nConfiguration also saved to: {output_file}")
    print("\nNext steps:")
    print(f"  1. Copy the above code to src/utils.py")
    print(f"  2. Run: python run_anomalydino.py --dataset {dataset_name} --data_root {data_root}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan dataset structure and generate registration code for AnomalyDINO"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the symlink dataset structure"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name for the dataset (used with --dataset argument)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    register_dataset(args.data_root, args.dataset_name)
