# util/config_loader.py

import yaml
import os
from argparse import Namespace
from typing import Dict, Any

def load_config(args: Namespace) -> Namespace:
    """
    加载配置文件并处理动态路径
    """
    config_path = getattr(args, 'config', None)
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # 加载 YAML 配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 计算动态变量
    config_dict = compute_dynamic_variables(config_dict)
    
    # 替换模板
    config_dict = resolve_templates(config_dict)
    
    # 将配置字典转换为 Namespace
    config_namespace = Namespace(**config_dict)
    
    # 合并参数
    merged_dict = {**vars(config_namespace), **vars(args)}
    merged_dict = {k: v for k, v in merged_dict.items() if v is not None}
    
    return Namespace(**merged_dict)


def compute_dynamic_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算动态变量
    """
    config = config.copy()
    
    # 根据 use_ray_enhancement 决定 module_part
    use_ray_enhancement = config.get('use_ray_enhancement', False)
    adapter_module = config.get('adapter_module', 'unknown')
    
    if use_ray_enhancement:
        config['module_part'] = adapter_module
    else:
        config['module_part'] = 'no_ray_enhancement'
    
    # 添加实验标识
    config['exp_id'] = f"{config['module_part']}_{config.get('activation', 'default')}"
    
    return config


def resolve_templates(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析模板字符串
    """
    config = config.copy()
    
    # 需要解析的模板字段
    template_fields = ['save_path_template', 'log_directory_template']
    
    for template_field in template_fields:
        if template_field in config:
            # 获取模板字符串
            template = config[template_field]
            
            # 替换所有变量
            for key, value in config.items():
                placeholder = f"{{{key}}}"
                if placeholder in template:
                    template = template.replace(placeholder, str(value))
            
            # 生成最终字段名（去掉_template后缀）
            field_name = template_field.replace('_template', '')
            config[field_name] = template
            
            # 删除模板字段
            del config[template_field]
    
    return config
# #!/usr/bin/env python3
# # util/config_loader.py

# import yaml
# import os
# from argparse import Namespace
# from typing import Dict, Any
# import re

# def load_config(args: Namespace) -> Namespace:
#     """
#     加载配置文件并解析其中的变量
    
#     Args:
#         args: 包含 config 文件路径的 argparse Namespace
    
#     Returns:
#         更新后的 Namespace，包含所有配置
#     """
#     # 获取配置文件路径
#     config_path = getattr(args, 'config', None)
#     if not config_path or not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found: {config_path}")
    
#     # 加载 YAML 配置
#     with open(config_path, 'r', encoding='utf-8') as f:
#         config_dict = yaml.safe_load(f)
    
#     # 如果 YAML 文件中有变量占位符，解析它们
#     config_dict = resolve_config_variables(config_dict)
    
#     # 将配置字典转换为 Namespace
#     config_namespace = Namespace(**config_dict)
    
#     # 合并原始 args 和新配置
#     # 优先保留命令行参数，然后才是配置文件中的参数
#     merged_dict = {**vars(config_namespace), **vars(args)}
#     merged_args = Namespace(**merged_dict)
    
#     return merged_args


# def resolve_config_variables(config: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     解析配置中的变量占位符
    
#     Args:
#         config: 原始配置字典
    
#     Returns:
#         解析后的配置字典
#     """
#     if not config:
#         return config
    
#     # 深拷贝配置
#     resolved = config.copy()
    
#     # 定义需要替换的字段
#     variable_fields = ['save_path', 'log_directory']
    
#     for field in variable_fields:
#         if field in resolved and isinstance(resolved[field], str):
#             # 使用 format 方法替换变量
#             try:
#                 resolved[field] = resolved[field].format(**config)
#             except KeyError as e:
#                 print(f"Warning: Variable {e} not found in config for field {field}")
#                 # 如果变量不存在，保持原样
#                 pass
#             except Exception as e:
#                 print(f"Warning: Failed to format {field}: {e}")
    
#     return resolved


# def save_config(config: Dict[str, Any], path: str):
#     """
#     保存配置到文件
    
#     Args:
#         config: 配置字典
#         path: 保存路径
#     """
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, 'w', encoding='utf-8') as f:
#         yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


# if __name__ == "__main__":
#     # 测试代码
#     test_args = Namespace(config="test_config.yaml")
#     loaded = load_config(test_args)
#     print("Loaded config:", loaded)