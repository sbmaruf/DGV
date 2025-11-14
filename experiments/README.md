# data.yaml

Each of the line comes with a megatron tokenized bin, idx data format. 

```yaml
- ar_Filtered-Ara-Translated-Ultra-Chat_000_text_document_dc=689278_sc=689278_tc=466898118.bin
- ar_Filtered-Ara-Translated-Ultra-Chat_000_text_document_dc=689278_sc=689278_tc=466898118.idx
- ar_Filtered-Ara-Translated-Ultra-Chat_001_text_document_dc=667637_sc=667637_tc=468696206.bin
- ar_Filtered-Ara-Translated-Ultra-Chat_001_text_document_dc=667637_sc=667637_tc=468696206.idx
...
```

The naming convention is below, 

`{LANG}_{DATA_SOURCE_NAME}_{SHARD_ID}_text_document_dc={NUMBER_OF_DOC}_sc={NUMBER_OF_SENTENCES}_tc={NUMBER_OF_TOKEN}.{bin/idx}`


It is actually not ideal to have the meta information in the file names, but it helps all the time to quickly skim through the `*jsonl` files. However as the production moves forward, standardizing this more would be nice feature.

You will mainly edit the `data_ratio.yaml` file in a brainstorming session and decide how many tokens the data source has and how many tokens you want to use from them to train the model. 

# Data Orchestration Process

## Overview

The data orchestration process calculates iterator probabilities for training data based on domain ratios, language selection probabilities, and total token budgets. This is handled by the `prepare_iterator_prob.sh` script.

## Running the Script

To orchestrate the data and calculate iterator probabilities, run:

```bash
bash experiments/dummy/prepare_iterator_prob.sh > experiments/dummy/meta/out.txt
```

This script will generate the iterator configuration and human-readable summaries of the data distribution.

## What It Does

The `prepare_iterator_prob.sh` script:

1. **Checks for data availability** - Verifies that `experiments/dummy/conf/data.yaml` exists (downloads data if needed)
2. **Calculates iterator probabilities** - Runs `src/data/calc_iterator_prob.py` with the following configurations:
   - Uses data paths from `data.yaml`
   - Applies domain ratios from `data_ratio.yaml`
   - Applies language selection probabilities from `lang_prob.yaml`
   - Respects exclusions from `exclude_iterator.yaml`
3. **Generates outputs**:
   - Iterator shell script: `experiments/dummy/conf/iter.sh`
   - Human-readable CSV summaries: `experiments/dummy/meta/` directory

## Configuration Files

### Required Configuration Files

- **`data.yaml`**: Lists all available data files with their paths
- **`data_ratio.yaml`**: Specifies how many tokens to use from each data source
- **`lang_prob.yaml`**: Language-specific sampling probabilities
- **`exclude_iterator.yaml`**: Data sources to exclude from training

### Parameters

- `--total-token 3_000_000_000_000`: Total token budget for training (3 trillion tokens in this example)
- `--prefix-for-file-path "$BIN_IDX_PATH"`: Environment variable prefix for data file paths, need this to plug in inside a megatron arguments.
- `--human-readable-export-type "csv"`: Export format for summaries

## Output Files

After running the script, you'll find:

- **`experiments/dummy/conf/iter.sh`**: Shell script with the calculated iterator configuration
- **`experiments/dummy/meta/out.txt`**: Verbose output from the calculation process
- **`experiments/dummy/meta/data_ratio.csv`**: CSV summary of domain ratios
- **`experiments/dummy/meta/data_summary.csv`**: CSV summary of the overall data distribution

## Workflow

1. Prepare your configuration files (`data.yaml`, `data_ratio.yaml`, `lang_prob.yaml`, `exclude_iterator.yaml`)
2. Run `bash experiments/dummy/prepare_iterator_prob.sh > experiments/dummy/meta/out.txt`
3. Review the generated CSV files in `experiments/dummy/meta/` to verify the data distribution
4. Use the generated `iter.sh` script in your training pipeline
