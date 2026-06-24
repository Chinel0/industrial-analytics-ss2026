import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('IA_Supply_management.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
for i in [156, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181]:
    c = cells[i-1]
    src = ''.join(c['source'])
    print(f'=== CELL {i} [{c["cell_type"]}] ===')
    print(src)
    print()
