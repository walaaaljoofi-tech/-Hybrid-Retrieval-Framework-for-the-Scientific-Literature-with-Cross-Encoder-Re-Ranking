dataset_name = "scifact"
dataset_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
data_path = util.download_and_unzip(dataset_url, "datasets")

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

print(f"#docs = {len(corpus)}, #queries = {len(queries)}, #qrels = {len(qrels)}")
