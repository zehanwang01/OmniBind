# OmniBind






## File structure
```
-assets
      [demo samples, including images, audios and point clouds]
-bpe
      [bpe_simple_vocab_16e6.txt.gz for imagebind]
-checkpoints
      [pretrained weights for OmniBind and experts]
-omni_model
      -eva_clip                     [part of eva_clip code]
      -laion_clap                   [part of laion_clap code]
      -Uni3D                        [part of Uni3D code]
      -wavcaps                      [part of wavcaps code]
      -transformer_430              [part of transformer v4.30.2 code]
      -tokenizers_013               [part of tokenizers v0.13 code]
      projector.py                  [the projector of OmniBind]
      router.py                     [the projector of OmniBind]
      experts.py                    [base feature extractors]
      omni_spaces.py                [combine router and experts together]
      omni_utils.py                 [useful functions]
      paths.py                      [paths of experts repo and weights]
      type.py                       [modality types]
```

## Usage

### 1. preparing enviornments
Install pytorch and other 3rd party dependencies.
Note that SigLip and laion-clap has environment conflict on transformers. We choose the 4.37.2 transformer version for siglip and built parts of 4.30.2 into the repository for CLAP.
To install ImageBind in this environment, you should clone the [ImageBind repository](https://github.com/facebookresearch/ImageBind), then follow the guide in the repo.
Make sure `libgeos++-dev` is installed in your environment(`sudo apt install libgeos++-dev`), otherwise the you may fail to `pip install cartorpy` for imagebind.
See `preparation.sh` for more details, or just execute the file.
```shell
chmod +x preparation.sh
bash ./preparation.sh
```

### 2. download checkpoints
We have made minor changes to the code of `CLAP`, `Wavcaps` and `Uni3D` to make them better initialized in `OmniBind`, and the relevant code is included in the `omni_model` directory.

The encoder and projectors of OmniBind have been included in the checkpoint we prepared([Huggingface OmniBind](https://huggingface.co/Viglong/OmniBind)).

You can execute the following script to set `checkpoints`.

```shell
git lfs clone https://huggingface.co/Viglong/OmniBind
mv OmniBind/checkpoints ./checkpoints
rm -r OmniBind
```

Next, check that the path in file `omni_model/paths.py` is correct.

The final structure of `checkpoints` should be like this:
```
-checkpoints
    -clap                                               [pretrained weights for CLAP]
        630k_clap_fullset_fusion.pt
        music_speech_audioset_epoch_15_esc_89.98.pt
    -projs                                              [space projectors and routers]
        base.pt
        large.pt
        full.pt
    -uni3d-g                                            [pretrained weights for Uni3D]
        -lvis/model.pt
        -mnet40/model.pt
        -scanobjnn/model.pt
    -wavcaps                                            [pretrained weights for wavcaps]
        HTSAT-BERT-PT.pt
        HTSAT-BERT-FT-AudioCaps.pt
        HTSAT-BERT-FT-Clotho.pt
    EVA02_CLIP_E_psz14_plus_s9B.pt                      [pretrained weights for EVA_CLIP_E14p]
    
```

### 3. Inference

Extract and compare embeddings in OmniBind:


```python
from omni_model.omni_space import *
from safetensors.torch import load_model
a = OmniBind_Large(pretrained=True)
load_model(a, 'checkpoints/large.safetensors')
a = a.cuda()
with torch.no_grad():
    aud = a.emb_audios(['assets/train.wav', 'assets/toilet.wav'])
    img = a.emb_images(['assets/train.jpeg', 'assets/toilet.jpeg'])
    txt = a.emb_texts(['a photo of train', 'a photo of toilet'])
    pc = a.emb_texts(['assets/train.npy', 'assets/toilet.npy'])
print(aud.shape, img.shape, txt.shape, pc.shape)
print(aud@img.T)
print(aud@txt.T)
print(aud@pc.T)
print(img@txt.T)
print(img@pc.T)
print(txt@pc.T)

```
