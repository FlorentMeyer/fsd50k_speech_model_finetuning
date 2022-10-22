# fsd50k_speech_model_fine_tuning

MWE of fine-tuning a Transformer-based speech embedder (e.g. [wav2vec 2.0](https://arxiv.org/abs/2006.11477)) on a subset of FSD50K using `pytorch_lightning` and HuggingFace `transformers`. 

Please refer to this executable [Colab notebook](https://colab.research.google.com/drive/1NddRCV1BtwgK6tvnylkLHY8d7t4OhAEw?usp=sharing) importing the code from this repo as well as a 500-element subset of the original FSD50K dataset for a concrete train+test example.

Note: intended as an editable incentive for jumping into FSD50K and the Pytorch-Lightning+HuggingFace framework, and as a showcase for an end-of-studies project -- choices have been made and some logic has been altered to (greatly) reduce the size of the original code.

Attribution and licenses:
- [The FSD50K dataset](https://zenodo.org/record/4060432) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- The 500-element subset used here only includes [CC0 1.0](http://creativecommons.org/publicdomain/zero/1.0/) audio samples