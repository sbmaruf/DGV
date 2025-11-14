###
# bash experiments/dummy/prepare_iterator_prob.sh > experiments/dummy/meta/out.txt 
#
###


if [ ! -f "experiments/dummy/conf/data.yaml" ]; then
    # load and download the data files. 
    echo "No data"
fi


python src/data/calc_iterator_prob.py \
 --prefix-paths-from-json "experiments/dummy/conf/data.yaml" \
 --domain-ratio-from-json "experiments/dummy/conf/data_ratio.yaml" \
 --lang-select-prob-json "experiments/dummy/conf/lang_prob.yaml" \
 --total-token 3_000_000_000_000 \
 --exclude-iterator-json "experiments/dummy/conf/exclude_iterator.yaml" \
 --prefix-for-file-path "\$BIN_IDX_PATH" \
 --export-script "experiments/dummy/conf/iter.sh" \
 --human-readable-export-type "csv" \
 --human-readable-export-path "experiments/dummy/meta/" \
 --verbose