import json, re

NB = r"c:\Users\nweke\OneDrive\Desktop\Industrial_Analytics\industrial-analytics-ss2026\SS2026_IBS6_IA_GR04_Chinelo_Nweke_Machine_Learning.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

def is_banner_line(line):
    s = line.strip()
    # print("=" * 60) or print("=" * N)
    if re.match(r'^print\(["\']=+["\'] \* \d+\)', s):
        return True
    # print("-" * 60)
    if re.match(r'^print\(["\'-]+["\'] \* \d+\)', s):
        return True
    # print("PHASE 3: HYPERPARAMETER TUNING (GridSearchCV)")  -- all-caps banner labels
    if re.match(r'^print\(["\'](PHASE|KNN REGRESSION|LINEAR REGRESSION|FINAL MODEL)[^"\']*["\']\)', s):
        return True
    return False

removed = 0
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        original = cell["source"]
        filtered = [line for line in original if not is_banner_line(line)]
        removed += len(original) - len(filtered)
        cell["source"] = filtered

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Removed {removed} banner lines.")
