"""Synthetic records for engineering verification; contains no challenge data."""
import csv
from pathlib import Path
import sys


def make(root, n=100):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for mode in ('train', 'test'):
        sources = {1: [], 2: [], 3: []}
        truth = []
        for i in range(n):
            country = 'France' if mode == 'test' else ('India' if i % 2 else 'US')
            name = f'श्री गणेश दुकान {i}' if i % 2 else f'Étoile bakery branch {i}'
            address = '' if i % 11 == 0 else f'{i+200} Rue du Marché'
            sid = f'S1-{i:04d}'
            sources[1].append([sid, name, address, country])
            links = []
            for source in (2, 3):
                tid = f'S{source}-{i:04d}'
                if i % 5:
                    sources[source].append([tid, name, address, country])
                    links.append(tid)
                else:
                    sources[source].append([tid, f'Unrelated industrial shop {i}', f'{i+9000} Highway', country])
                sources[source].append([f'S{source}-noise-{i:04d}', f'Foreign corporation {i}', f'{i+8000} Avenue', country])
            truth.append([sid, ','.join(links)])
        for source, rows in sources.items():
            with (root/f'{mode}_source{source}.tsv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream, delimiter='\t')
                writer.writerow(['entity_id', 'business_name', 'business_address', 'country'])
                writer.writerows(rows)
        if mode == 'train':
            with (root/'train_ground_truth.tsv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream, delimiter='\t')
                writer.writerow(['source1_entity_id', 'matched_entity_ids'])
                writer.writerows(truth)

if __name__ == '__main__':
    make(sys.argv[1] if len(sys.argv)>1 else '.fixture/data')
