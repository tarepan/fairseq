# FAIRSEQ S<sup>2</sup>
Speech synthesis with fairseq.

## Features

- Autoregressive and non-autoregressive models
- Multi-speaker synthesis
- Audio preprocessing (denoising, VAD, etc.) for less curated data
- Automatic metrics for model development
- Similar data configuration as [S2T](../speech_to_text/README.md)

## Models

### Input

#### Type

| name   | type          | model                            | FS2 duration extraction      |
|--------|---------------|----------------------------------|------------------------------|
| `Char` | character     | -                                | ?                            |
| `g2pE` | phoneme       | g2pE                             | Montreal Forced Aligner      |
| `espk` | phoneme       | Phonemizer w/ espeak-ng1 backend | Montreal Forced Aligner      |
| `Unit` | acoustic unit | `hubert_base_ls960`              | repeat counting [^fs2_input] |

[^fs2_input]: "For discovered units, we extract framelevel units using a Base HuBERT model trained on LibriSpeech and collapse consecutive units of the same kind. We use the run length of identical units before collapsing as target duration for FastSpeech 2 training." from original paper

#### Dataset
- LJSpeech
- VCTK
- CommonVoice

### Network
- feat2spec
  - `TFM`: Transformer TTS
  - `FS2`: FastSpeech 2
- spec2wave
  -HiFiGAN

## Examples
- [Single-speaker synthesis on LJSpeech](docs/ljspeech_example.md)
- [Multi-speaker synthesis on VCTK](docs/vctk_example.md)
- [Multi-speaker synthesis on Common Voice](docs/common_voice_example.md)


## Original Paper
[![Paper](http://img.shields.io/badge/paper-arxiv.2109.06912-B31B1B.svg)][paper]  
<!-- https://arxiv2bibtex.org/?q=2109.06912&format=bibtex -->
```
@misc{2109.06912,
Author = {Changhan Wang and Wei-Ning Hsu and Yossi Adi and Adam Polyak and Ann Lee and Peng-Jen Chen and Jiatao Gu and Juan Pino},
Title = {fairseq S^2: A Scalable and Integrable Speech Synthesis Toolkit},
Year = {2021},
Eprint = {arXiv:2109.06912},
}
```

[paper]:https://arxiv.org/abs/2109.06912

### Citation
Please cite as:
```
@article{wang2021fairseqs2,
  title={fairseq S\^{} 2: A Scalable and Integrable Speech Synthesis Toolkit},
  author={Wang, Changhan and Hsu, Wei-Ning and Adi, Yossi and Polyak, Adam and Lee, Ann and Chen, Peng-Jen and Gu, Jiatao and Pino, Juan},
  journal={arXiv preprint arXiv:2109.06912},
  year={2021}
}

@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
