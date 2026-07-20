import os
import json
import re

brain_dir = r"C:\Users\omnig\.gemini\antigravity\brain"
output_file = r"J:\UnivHm_NextJS\Al-Kutub-Automator\search_results.txt"

results = []

for root, dirs, files in os.walk(brain_dir):
    for name in files:
        if name.endswith('.jsonl'):
            path = os.path.join(root, name)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if any(kwd in line for kwd in ['filedn', 'lB8bgDEsPWLVIlRdbAmCPRm']):
                            results.append({
                                'file': path,
                                'line': i,
                                'content': line.strip()[:1000] # truncate line length for safety
                            })
            except Exception as e:
                results.append({
                    'file': path,
                    'error': str(e)
                })

with open(output_file, 'w', encoding='utf-8') as out:
    for res in results:
        out.write(f"File: {res.get('file')}\nLine: {res.get('line')}\n")
        if 'error' in res:
            out.write(f"Error: {res['error']}\n")
        else:
            out.write(f"Content: {res['content']}\n")
        out.write("-" * 80 + "\n")

print(f"Done. Found {len(results)} matches.")
