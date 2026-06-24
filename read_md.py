import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open(r'c:\Users\nweke\OneDrive\Desktop\Industrial_Analytics\industrial-analytics-ss2026\SS2026_IBS6_IA_GR04_Chinelo_Nweke_Machine_Learning.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown':
        src = ''.join(c['source'])
        print(f"=== CELL {i} id={c['id']} ===")
        print(src)
        print()
