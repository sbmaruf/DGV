# data.yaml

Each of the line comes with a megatron tokenized bin, idx data format. 

```
- ar_Filtered-Ara-Translated-Ultra-Chat_000_text_document_dc=689278_sc=689278_tc=466898118.bin
- ar_Filtered-Ara-Translated-Ultra-Chat_000_text_document_dc=689278_sc=689278_tc=466898118.idx
- ar_Filtered-Ara-Translated-Ultra-Chat_001_text_document_dc=667637_sc=667637_tc=468696206.bin
- ar_Filtered-Ara-Translated-Ultra-Chat_001_text_document_dc=667637_sc=667637_tc=468696206.idx
...
```

The naming convention is below, 

{LANG}_{DATA_SOURCE_NAME}_{SHARD_ID}_text_document_dc={NUMBER_OF_DOC}_sc={NUMBER_OF_SENTENCES}_tc={NUMBER_OF_TOKEN}.{bin/idx}


It is actually not ideal to have the meta information in the file names, but it helps all the time to quickly skim through the `*jsonl` files. However as the production moves forward, standardizing this more would be nice feature.

You will mainly edit the `data_ratio.yaml` file in a brainstorming session and decide how many tokens the data source has and how many tokens you want to use from them to train the model. 


