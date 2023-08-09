import copy
import json

from recognizer.utils.constants import ROOT_DIR

DATA_DIR = ROOT_DIR / "data" / "data"

with open(DATA_DIR / "preds.jsonl", "r") as f:
    json_lines = f.readlines()

items = [json.loads(line) for line in json_lines]

dynamo_items = []

for item in items:
    user_id = item["input"]["userId"]
    ranked_promos = zip(item["output"]["recommendedItems"], item["output"]["scores"])

    dynamo_item = {
        "Item":{
            "userId":{"S": user_id},
            "items":{"L":[]}
        }
    }
    for promo_slug, score in ranked_promos:
        dynamo_item["Item"]["items"]["L"].append(
            {
                "M": {
                    "itemId": {"S":promo_slug},
                    "score": {"N": str(score)}
                }
            }
        )
    dynamo_items.append(copy.deepcopy(dynamo_item))

lines = [json.dumps(dynamo_item).replace(" ", "") for dynamo_item in dynamo_items]

with open(DATA_DIR / "parsed.json", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")
