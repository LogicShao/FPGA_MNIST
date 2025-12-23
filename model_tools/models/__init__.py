"""
模型注册系统
支持动态加载和切换不同的神经网络模型
"""

from .SimpleMLP import SimpleMLP
from .TinyLeNet import TinyLeNet

# 模型注册表
MODEL_REGISTRY = {
    'SimpleMLP': {
        'class': SimpleMLP,
        'type': 'mlp',
        'description': '简单的2层MLP (784->32->10)',
        'input_shape': (1, 28, 28),
        'params': '~25K',
    },
    'TinyLeNet': {
        'class': TinyLeNet,
        'type': 'cnn',
        'description': 'Tiny-LeNet CNN (C1->S2->C3->S4->FC)',
        'input_shape': (1, 28, 28),
        'params': '~11K',
    },
}


def get_model(name):
    """
    根据名称获取模型类
    Args:
        name: 模型名称，如'SimpleMLP'或'TinyLeNet'
    Returns:
        模型类
    """
    if name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available models: {available}")

    model_info = MODEL_REGISTRY[name]
    if model_info['class'] is None:
        raise ValueError(f"Model {name} is not available")

    return model_info['class']


def get_model_info(name):
    """获取模型详细信息"""
    if name not in MODEL_REGISTRY:
        return None
    return MODEL_REGISTRY[name]


def list_models():
    """列出所有可用模型"""
    print("Available models:")
    print("-" * 70)
    for name, info in MODEL_REGISTRY.items():
        if info['class'] is not None:
            print(f"  {name:15} | {info['type']:4} | {info['description']}")
            print(f"  {'':15} | Params: {info['params']}")
    print("-" * 70)


__all__ = ['SimpleMLP', 'TinyLeNet', 'get_model', 'get_model_info', 'list_models', 'MODEL_REGISTRY']
