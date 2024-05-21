The pre-trained model can be downloaded here: https://drive.google.com/file/d/1xjWRr1GDhwQZxHnx2fQNA7P5_W_iBOZJ/view?usp=sharing
To use the text-only AEC model, please refer to ``text_only_main.py``. You only need to prepare your asr transcript and groundtruth data.
To use the crossmodal AEC model, please refer to ``crossmodal_main.py``. You need to generate your own discrete speech units first before using the code.
I did not include models for continuous audio features. You can simply modify the model architecture (e.g., rescaling, downsampling) and integrate audio feature extraction in ``crossmodal_main.py``.
