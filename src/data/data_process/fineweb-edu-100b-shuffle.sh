git clone https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle
cd fineweb-edu-100b-shuffle
find . -type f  -name "*.parquet" -print0 | xargs -0 -I {} sh -c 'python3 -c "import pandas as pd; df = pd.read_parquet(\"{}\"); df.to_json(\"{}\".replace(\".parquet\", \".jsonl\"), orient=\"records\", lines=True)"'
