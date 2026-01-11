from setuptools import setup, find_packages

setup(
    name='vllm_add_dummy_model',
    version='0.1',
    packages=find_packages(include=["vllm_add_dummy_model*", "model_utils*"]),
    entry_points={
        'vllm.general_plugins': [
            "register_dummy_model = vllm_add_dummy_model.prm_model:register"
        ]
    }
)