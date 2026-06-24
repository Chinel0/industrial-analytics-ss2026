import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('IA_Supply_management.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
for i in [182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203]:
    c = cells[i-1]
    src = ''.join(c['source'])
    print(f'=== CELL {i} [{c["cell_type"]}] ===')
    print(src)
    print()
