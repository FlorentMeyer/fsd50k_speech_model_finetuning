from setuptools import setup

setup(
    name='FSD50K Speech Model Fine-tuning',
    description='MWE of fine-tuning a Transformer-based speech embedder (e.g. wav2vec 2.0) on a subset of FSD50K using pytorch_lightning and HuggingFace transformers.',
    author='Florent Meyer',
    author_email='florentmeyer@outlook.fr',
    packages=['fsd50k_speech_model_finetuning'],
)