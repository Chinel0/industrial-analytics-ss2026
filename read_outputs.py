import json, sys
sys.stdout.reconfigure(encoding='utf-8')
path = r'c:\Users\nweke\OneDrive\Desktop\Industrial_Analytics\industrial-analytics-ss2026\SS2026_IBS6_IA_GR04_Chinelo_Nweke_Machine_Learning.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and c.get('outputs'):
        print(f"=== CELL {i} ===")
        for out in c['outputs']:
            if out.get('output_type') in ('stream', 'execute_result', 'display_data'):
                text = out.get('text', out.get('data', {}).get('text/plain', ''))
                if isinstance(text, list):
                    text = ''.join(text)
                print(text[:2000])
        print()
