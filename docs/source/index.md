# SuPar

[![build](https://img.shields.io/github/workflow/status/yzhangcs/parser/build?style=flat-square)](https://github.com/yzhangcs/parser/actions)
[![docs](https://readthedocs.org/projects/parser/badge/?version=latest&style=flat-square)](https://parser.readthedocs.io/en/latest)
[![release](https://img.shields.io/github/v/release/yzhangcs/parser?style=flat-square)](https://github.com/yzhangcs/parser/releases)
[![downloads](https://img.shields.io/github/downloads/yzhangcs/parser/total?style=flat-square)](https://pypistats.org/packages/supar)
[![LICENSE](https://img.shields.io/github/license/yzhangcs/parser?style=flat-square)](https://github.com/yzhangcs/parser/blob/master/LICENSE)

A Python package designed for structured prediction, including reproductions of many state-of-the-art syntactic/semantic parsers (with pretrained models for more than 19 languages),
and highly-parallelized implementations of several well-known structured prediction algorithms.[^1]

```{toctree}
:maxdepth: 2
:caption: Content

models/index
structs/index
modules/index
utils/index
refs
```

## Indices and tables

* [](genindex)
* [](modindex)
* [](search)

[^1]: The implementations of structured distributions and semirings are heavily borrowed from [torchstruct](https://github.com/harvardnlp/pytorch-struct) with some tailoring.
