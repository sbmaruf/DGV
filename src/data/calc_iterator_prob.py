import os
import yaml
import glob
import json
import csv
import copy
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-prefix-paths', default=None, type=str,
                       help='Glob path to folder where all the bin, idx files are.')
    parser.add_argument('--prefix-paths', '--prefix-paths-from-json', default=None, type=str,
                       help='File names listed in a json or yaml file.')
    parser.add_argument('--domain-ratio', '--domain-ratio-from-json', type=str, required=True,
                       help='data multipliers defined in a json or yaml file.')
    parser.add_argument('--lang-select-prob', '--lang-select-prob-json', type=str, required=True,
                       help='Path to a json or yaml file that indicates the lang ratio')
    parser.add_argument('--exclude-iterator', '--exclude-iterator-json', type=str, required=False, default=None,
                       help='(Optional) Path to a json or yaml file that list the data shards to exclude')
    parser.add_argument('--total-token', type=int, required=True,
                       help='Total token to be sampled.')
    parser.add_argument('--verbose', action='store_true',
                       help='Print additional information')
    parser.add_argument('--export-script', type=str,
                       help='Export output to this file in megatron format.')
    parser.add_argument('--prefix-for-file-path', type=str,
                       help='Add additional prefix to the file path.')
    parser.add_argument('--human-readable-export-type', type=str, default="csv",
                        choices=["csv", "json"],
                        help='Output file to save the data.')
    parser.add_argument('--human-readable-export-path', type=str, default="./",
                        help='Output file to save the data.')
    
    args = parser.parse_args()
    if args.source_prefix_paths is not None:
        assert args.prefix_paths is None
    if args.prefix_paths is not None:
        assert args.source_prefix_paths is None
    if args.prefix_for_file_path is not None:
        if args.prefix_for_file_path.endswith("/"):
            args.prefix_for_file_path = args.prefix_for_file_path[:-1]

    return args

def load_file(file_name):
    """ loads a json or yaml file based on the file extension and returns a dict """
    if file_name.lower().endswith('.json'):
        return json.load(open(file_name, 'r'))
    elif file_name.lower().endswith(('.yaml', '.yml')):
        return yaml.safe_load(open(file_name, 'r'))
    else:
        raise NotImplementedError(f"File extension for {file_name} not recognized!")

def normalize(in_dict):
    """ normalizes the values in a dictionary """
    total = sum(in_dict.values())
    for k, v in in_dict.items():
        in_dict[k] = v/total
    return in_dict

def initialize_dict(in_dict):
    """ initializes data structure to track datasets that can be nested up to two times, e.g:
    {
        "lang":
            "domain":
                "dataset": epochs
    }
    or
    {
        "lang":{
            "dataset": epocs
        }
    }
    or
    {
        "dataset": epochs
    }
    """
    out_dict = copy.deepcopy(in_dict)
    for k, v in in_dict.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                if isinstance(v1, dict):
                    for k2 in v1.keys():
                        out_dict[k][k1][k2] = 0
                else:
                    out_dict[k][k1] = 0
        elif not isinstance(v, dict):
            out_dict[k] = 0
        else:
            raise NotImplementedError("Dict format not recognized. Should be dict nested up to two times")
    return out_dict


def format_lang_domain_dataset_dict(in_dict):
    """Format the numbers in the dictionary (language, domain, dataset), e.g:
    {
        "lang":
            "domain":
                "dataset": epochs
    }
    or
    {
        "lang":{
            "dataset": epocs
        }
    }
    or
    {
        "dataset": epochs
    }
    """
    out_dict = copy.deepcopy(in_dict)
    for k, v in in_dict.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                if isinstance(v1, dict):
                    for k2 in v1.keys():
                        if isinstance(out_dict[k][k1][k2], int):
                            out_dict[k][k1][k2] = f"{out_dict[k][k1][k2]:,}"
                        elif isinstance(out_dict[k][k1][k2], float):
                            out_dict[k][k1][k2] = f"{out_dict[k][k1][k2]:.4f}"
                        elif isinstance(out_dict[k][k1][k2], dict):
                            for k3 in out_dict[k][k1][k2].keys():
                                out_dict[k][k1][k2][k3] = f"{out_dict[k][k1][k2][k3]:,}"
                else:
                    if isinstance(out_dict[k][k1], int):
                           out_dict[k][k1] = f"{out_dict[k][k1]:,}"
                    elif isinstance(out_dict[k][k1], float):
                       out_dict[k][k1] = f"{out_dict[k][k1]:.4f}"

        elif not isinstance(v, dict):
            out_dict[k] = 0
        else:
            raise NotImplementedError("Dict format not recognized. Should be dict nested up to two times")
    return out_dict


def get_lang_dataset_in_dict(in_dict, lang, dataset):
    """ returns the value in a dictionary keyed by any combination of language, domain, and dataset """
    out_dict = copy.deepcopy(in_dict)

    if lang in in_dict: out_dict = out_dict[lang]
    if dataset in out_dict: out_dict = out_dict[dataset]
    if isinstance(out_dict, dict):
        for domain in out_dict.keys():
            if isinstance(out_dict[domain], dict) and dataset in out_dict[domain]:
                out_dict = out_dict[domain][dataset]
                break
    return out_dict

def accumulate_lang_dataset_in_dict(in_dict, lang, dataset, value):
    """ accumulates values in a dictionary keyed by any combination of language, domain, and dataset """
    assert lang in in_dict, f"{lang} not in {in_dict.keys()}"
    if lang in in_dict:
        if dataset in in_dict[lang]:
            in_dict[lang][dataset] += value
        else:
            for domain in in_dict[lang].keys():
                if isinstance(in_dict[lang][domain], dict) and dataset in in_dict[lang][domain]:
                    in_dict[lang][domain][dataset] += value
    else: in_dict[dataset] += value
    return in_dict

def load_or_create_meta_files(args):
    """ loads the files or creates the data structures """
    if args.source_prefix_paths is not None:
        source_prefix_paths = glob.glob(args.source_prefix_paths)
    else:
        source_prefix_paths = load_file(args.prefix_paths)
    dataset_dict = load_file(args.domain_ratio)
    dataset_percentage_dict = initialize_dict(dataset_dict)
    dataset_token_dict = initialize_dict(dataset_dict)

    lang_ratio_dict = normalize(load_file(args.lang_select_prob))
    exclude_iterator_list = []
    if args.exclude_iterator is not None:
        exclude_iterator_list = load_file(args.exclude_iterator)['exclude_iterator_name']
    
    exclude_iterator_list = sorted([ exclude_iterator.replace(".bin", "") for exclude_iterator in exclude_iterator_list if exclude_iterator.endswith(".bin")])
    source_prefix_paths = sorted([ source_prefix_path.replace(".bin", "") for source_prefix_path in source_prefix_paths if source_prefix_path.endswith(".bin")])
    return (
        source_prefix_paths, 
        dataset_dict, 
        dataset_percentage_dict, 
        dataset_token_dict, 
        lang_ratio_dict, 
        exclude_iterator_list
    )

def hasdomain(_dict):
    for k, v in _dict.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                if isinstance(v1, dict):
                    for k2 in v1.keys():
                        if isinstance(v1[k2], dict): return True
                        else: return False
                else: return False
        else: return False
    return False
    
if __name__ == "__main__":
    args = get_args()

    # ~ load files
    (source_prefix_paths, 
     dataset_dict, 
     dataset_percentage_dict, 
     dataset_token_dict,
     lang_ratio_dict, 
     exclude_iterator_list) = load_or_create_meta_files(args)

    # ~ data_dist_by_lang dict tracks (by language) the number of desired tokens per data shard in that language
    data_dist_by_lang, tot_token_by_lang, tot_sampled_token_by_lang = {}, {}, {}
    
    # ~ collect token statistics per language/shard before and after sampling
    if args.verbose: print("\n\nFile Found ...")
    for idx, source_prefix_path in enumerate(source_prefix_paths):
        source_prefix_path = os.path.basename(source_prefix_path)
        
        if source_prefix_path in exclude_iterator_list: continue
        if args.verbose: print(f"\t{source_prefix_path}")
        try:
            # extract the number of tokens in this shard
            dc = int(source_prefix_path.split("dc=")[1].split("_")[0])
            sc = int(source_prefix_path.split("sc=")[1].split("_")[0])
            tc = int(source_prefix_path.split("tc=")[1].split("_")[0])
        except:
            raise ValueError(f"File name {source_prefix_path} not in the expected format.")
        
        lang = source_prefix_path.split("_")[0]
        dataset = source_prefix_path.split("_")[1]
        # extract the dataset/domain multiplier
        dataset_multiplier = get_lang_dataset_in_dict(dataset_dict, lang, dataset)

        # skip this shard if sampling epoch is 0
        if dataset_multiplier == 0: continue

        # skip this shard if this domain is not listed in domain_multipliers
        if isinstance(dataset_multiplier, dict): continue
        # initialize for each language if this is first time seeing this language
        if lang not in data_dist_by_lang: data_dist_by_lang[lang] = []
        if lang not in tot_token_by_lang: tot_token_by_lang[lang] = 0
        if lang not in tot_sampled_token_by_lang: tot_sampled_token_by_lang[lang] = 0
        # add this shard and desired token counts
        data_dist_by_lang[lang].append(
            { 
                "token_cnt_times_dataset_multiplier": tc * dataset_multiplier, 
                "iterator_name": source_prefix_path 
            }
        )
        tot_token_by_lang[lang] += tc
        tot_sampled_token_by_lang[lang] += (tc * dataset_multiplier)
    
    # ~ calculate the iterator selection probabilities
    iterator_selection_prob = []
    for lang, iterator_list in  data_dist_by_lang.items():
        tot_prob_covered = 0.0
        for meta_dict in iterator_list:
            iterator_tok_cnt = meta_dict['token_cnt_times_dataset_multiplier']
            iterator_name = meta_dict['iterator_name']
            dataset = iterator_name.split("_")[1]
            # compute the sampling probability of this shard as a fraction of the total dataset
            prob = iterator_tok_cnt/tot_sampled_token_by_lang[lang] * lang_ratio_dict[lang]
            dataset_multiplier = get_lang_dataset_in_dict(dataset_dict, lang, dataset)
            assert not isinstance(dataset_multiplier, dict)
            # store the prob, shard name, fraction of total dataset tokens, and number of tokens
            iterator_selection_prob.append(
                {
                    "probability": prob,
                    "iterator_name": iterator_name,
                    "total_token_to_be_sampled": int(prob*args.total_token),
                    "total_token_exists": iterator_tok_cnt//dataset_multiplier
                }
            )
            tot_prob_covered += prob
            # computes the percentage of the total dataset represented by each shard
            dataset_percentage_dict = accumulate_lang_dataset_in_dict(
                dataset_percentage_dict, lang, dataset, int(prob*args.total_token)
            ) # TODO: Review and come back, may be iterator_tok_cnt should be total_token_to_be_sampled (int(prob*args.total_token))
            dataset_token_dict = accumulate_lang_dataset_in_dict(
                dataset_token_dict, lang, dataset, iterator_tok_cnt//dataset_multiplier
            )
        assert abs(lang_ratio_dict[lang] - tot_prob_covered) < 1e-6
    
    if args.verbose: print("> Iterator selection probability.\n")
    lang_token = {k:0 for k, _ in lang_ratio_dict.items()}

    # ~ print/write out the iterator selection probabilites as a script to source.
    if args.export_script is not None:
        if not args.export_script.endswith(".sh"): args.export_script = args.export_script + ".sh"
        out_file_ptr = open(f"{args.export_script}", "w")
        out_file_ptr.write("DATA_PATH=( --data-path ")
    
    human_readable_export_data = [] 
    for meta_dict in iterator_selection_prob:
        prob = meta_dict['probability']
        iterator_name = meta_dict['iterator_name']
        total_token_to_be_sampled = meta_dict['total_token_to_be_sampled']
        total_token_exists = meta_dict['total_token_exists']
        lang = iterator_name.split("_")[0]
        lang_token[lang] += total_token_to_be_sampled
        if args.verbose: print(f"\t{prob} {os.path.basename(iterator_name)} {total_token_to_be_sampled:_} {total_token_exists:_} {total_token_to_be_sampled/total_token_exists:.2f}")
        __output_format = os.path.basename(iterator_name)#.replace('=', '\\=')
        if args.export_script is not None: out_file_ptr.write(f"\n\"{prob}\" \"{args.prefix_for_file_path}/{__output_format}\"")
        human_readable_export_data.append([prob, f"{args.prefix_for_file_path}/{__output_format}"])
    
    if args.human_readable_export_type == "csv":
        with open(os.path.join(args.human_readable_export_path, "data_ratio.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["probability", "iterator_name"])
            for row in human_readable_export_data:
                writer.writerow(row)
    elif args.human_readable_export_type == "json":
        with open(os.path.join(args.human_readable_export_path, "data_ratio.json"), "w") as f:
            json.dump(human_readable_export_data, f, indent=4)
    else:
        raise NotImplementedError(f"Export format {args.human_readable_export} not recognized!")
    
    if args.export_script is not None:
        out_file_ptr.write("\n)")
        out_file_ptr.close()
    
    # ~ print important statistics
    print(f"\n\n> Total tokens by language BEFORE SAMPLING.\n")
    for lang, token in tot_token_by_lang.items():
        print(f"\t\t{lang}: {token:,} ({token/1e9:.2f} B)")
    print(f"\n> Total tokens by language AFTER SAMPLING.\n")
    tot_sampled_token = 0
    for lang, token in tot_sampled_token_by_lang.items():
        tot_sampled_token += token
        print(f"\t\t{lang}: {token:,} ({token/1e9:.2f} B)")
    print(f"\n> Token percentages by language AFTER SAMPLING.\n")
    lang_percentage = dict()
    for lang, token in tot_sampled_token_by_lang.items():
        lang_percentage[lang] = token/tot_sampled_token
        print(f"\t\t{lang}: {token/tot_sampled_token*100.:.2f}%")

    print(f"\n> Out of {args.total_token} token, language wise token distribution.\n")
    for k, v in lang_token.items():
        print(f"\t\t{k}: {v:_} ({v/1e9:.2f} B)")
    
    domain_percentage = initialize_dict(dataset_percentage_dict)
    total_dataset_percentage = 0
    for key, val in dataset_percentage_dict.items():
        # if domain_dict is defined per language
        if isinstance(val, dict):
            lang = key
            for key1, val1 in val.items():
                if isinstance(val1, dict): # if val1 is a dict, then it is a key: domain_name, val1: dataset_name, sampled_token_count
                    domain = key1
                    total_domain_percentage, total_sampled_tokens, total_local_percentage, total_domain_tokens = 0, 0, 0, 0
                    for dataset in val1.keys():
                        dataset_percentage_dict[lang][domain][dataset] = dict(
                            global_ratio=round((dataset_percentage_dict[lang][domain][dataset]/args.total_token)*100, 4), 
                            local_ratio=round((dataset_percentage_dict[lang][domain][dataset]/lang_token[lang])*100, 4),
                            sampled_tokens=dataset_percentage_dict[lang][domain][dataset],
                            total_tokens=dataset_token_dict[lang][domain][dataset]
                        )
                        total_dataset_percentage += dataset_percentage_dict[lang][domain][dataset]['global_ratio']
                        total_domain_percentage += dataset_percentage_dict[lang][domain][dataset]['global_ratio']
                        total_sampled_tokens += dataset_percentage_dict[lang][domain][dataset]['sampled_tokens']
                        total_local_percentage += (dataset_percentage_dict[lang][domain][dataset]['sampled_tokens']/lang_token[lang])*100
                        total_domain_tokens += dataset_token_dict[lang][domain][dataset]
                    # domain_percentage[lang][domain] = total_domain_percentage      
                    domain_percentage[lang][domain] = {
                        "global": f"{total_domain_percentage:.4}",
                        "local": f"{total_local_percentage:.4}",
                        "sampled_tokens": f"{total_sampled_tokens:,}",
                        "total_tokens": f"{total_domain_tokens:,}"
                    }
                else: # else it is a key: dataset_name, val1: sampled_token_count
                    dataset = key1
                    dataset_percentage_dict[lang][dataset] = dict(
                        global_ratio=round((dataset_percentage_dict[lang][dataset]/args.total_token)*100, 4), 
                        local_ratio=round((dataset_percentage_dict[lang][dataset]/lang_token[lang])*100, 4),
                        sampled_tokens=dataset_percentage_dict[lang][dataset],
                        total_tokens=dataset_token_dict[lang][dataset]
                    )
                    total_dataset_percentage += dataset_percentage_dict[lang][dataset]['global_ratio']
                    total_domain_percentage = dataset_percentage_dict[lang][dataset]['global_ratio']
                    total_local_percentage = (dataset_percentage_dict[lang][dataset]['sampled_tokens']/lang_token[lang])*100
                    total_sampled_tokens = dataset_percentage_dict[lang][dataset]['sampled_tokens']
                    total_domain_tokens = dataset_token_dict[lang][dataset]
        else:
            assert len(tot_sampled_token_by_lang.keys()) == 1, "Domain dict is defined without languages!"
            lang = list(tot_sampled_token_by_lang.keys())[0]
            dataset, token = key, val
            dataset_percentage_dict[dataset] = dict(
                global_ratio=round(dataset_percentage_dict[dataset]/args.total_token*100, 4), 
                local_ratio=round(dataset_percentage_dict[dataset]/args.total_token*100, 4), 
                sampled_tokens=dataset_percentage_dict[dataset],
                total_tokens=dataset_token_dict[lang][domain][dataset]
            )
            total_dataset_percentage += dataset_percentage_dict[dataset]['global_ratio']
            raise NotImplementedError("This should not happen!")
    
    dataset_percentage_dict = format_lang_domain_dataset_dict(dataset_percentage_dict)
    print(f"\n> Dataset percentages.\n{json.dumps(dataset_percentage_dict, indent=4)}")
    if hasdomain(dataset_percentage_dict): 
        if args.human_readable_export_type == "csv":
            data_summary_file_path = os.path.join(args.human_readable_export_path, "data_summary.csv")
            with open(data_summary_file_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["language", "domain", "dataset", "global_ratio", "local_ratio", "sampled_tokens", "total_tokens"])
                for lang, dataset_dict in dataset_percentage_dict.items():
                    for domain, data in dataset_dict.items():
                        for dataset, values in data.items():
                            writer.writerow([lang, domain, dataset, values['global_ratio'], values['local_ratio'], values['sampled_tokens'], values['total_tokens']])
        elif args.human_readable_export_type == "json":
            data_summary_file_path = os.path.join(args.human_readable_export_path, "data_summary.json")
            with open(data_summary_file_path, "w") as f:
                json.dump(dataset_percentage_dict, f, indent=4)
        else:
            raise NotImplementedError(f"Export format {args.human_readable_export_type} not recognized!")
        print(f"\n> Domain percentages.\n{json.dumps(domain_percentage, indent=4)}")
    print(f"\n> Total data percentage (overestimate/underestimate due to precision): {total_dataset_percentage}")