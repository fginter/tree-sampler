# tree-sampler

Code to (attempt to) sample a treebank with a desired distribution of trees. The use case is that you have a large parsed treebank, with skewed distribution (maybe tending towards short, simple trees or something such) and then you have a treebank which you would like to enrich with same-looking data from the parsed treebank.

This would be the way to run:

```
zcat your_big_parsebank.conllu.gz | python3 sample.py --estimate-src-trees your_big_parsebank.conllu.gz --estimate-tgt-trees your_treebank.conllu > sampled_data.conllu
```

It does not matter whether the treebanks/parsebanks are .conllu.gz or .conllu, the program figures the compressin out from file names.
