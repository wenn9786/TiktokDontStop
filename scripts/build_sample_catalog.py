import json

# Collect every target ASIN the eval set will look for
needed = set()
with open("data/public_set.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        needed.add(sample["ground_truth"]["parent_asin"])

# Pull those rows out of the full catalog, plus pad with extra rows for realism
kept = []
padding = []
with open("data/catalog.jsonl") as f:
    for line in f:
        product = json.loads(line)
        asin = str(product["parent_asin"])
        if asin in needed:
            kept.append(line)
        elif len(padding) < 1000:
            padding.append(line)

with open("data/catalog_sample.jsonl", "w") as f:
    f.writelines(kept)
    f.writelines(padding)

print(f"Wrote {len(kept)} target products + {len(padding)} padding = {len(kept) + len(padding)} rows")