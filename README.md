The pre-trained model can be downloaded on Hugging Face: https://huggingface.co/YC-Li/Sequence-to-Sequence-ASR-Error-Correction

To use the text-only AEC model, please refer to ``text_only.py``. You only need to prepare your asr transcript and groundtruth data.

To use the crossmodal AEC model, please refer to ``crossmodal.py``. You need to generate your own discrete speech units first before using the code.

Models for continuous audio features are omitted. You can simply modify the model architecture (e.g., rescaling, downsampling) and integrate audio feature extraction in the model.

You may kindly cite

```
@inproceedings{li2024crossmodal,
  title={Crossmodal ASR error correction with discrete speech units},
  author={Li, Yuanchao and Chen, Pinzhen and Bell, Peter and Lai, Catherine},
  booktitle={2024 IEEE Spoken Language Technology Workshop (SLT)},
  pages={431--438},
  year={2024},
  organization={IEEE}
}
```
