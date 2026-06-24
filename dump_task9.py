import json, sys
sys.stdout.reconfigure(encoding='utf-8')
path = r'c:\Users\nweke\OneDrive\Desktop\Industrial_Analytics\industrial-analytics-ss2026\IA_Supply_management.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
for i in range(204, 222):
    c = cells[i-1]
    src = ''.join(c['source'])
    print(f'=== CELL {i} [{c["cell_type"]}] ===')
    print(src)
    print()
