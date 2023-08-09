import copy
import json

from recognizer.utils.constants import ROOT_DIR

DATA_DIR = ROOT_DIR / "data" / "data"

with open(DATA_DIR / "preds.jsonl", "r") as f_preds, open(DATA_DIR / "parsed2.json", "w") as f_parsed:
    for line in f_preds:
        item = json.loads(line)

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
        dynamo_item = json.dumps(dynamo_item).replace(" ", "")

        f_parsed.write(dynamo_item + "\n")
