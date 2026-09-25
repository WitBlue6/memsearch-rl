"""Reject entire samples with missing supporting sentences; retain an audit log."""
import json
from pathlib import Path

for name in ('hotpot_train_hf_2000', 'hotpot_dev_hf_200'):
    source = Path('data/raw') / f'{name}.json'
    rows = json.loads(source.read_text(encoding='utf-8'))
    kept, rejected = [], []

    for row in rows:
        context = dict(row['context'])
        bad = []
        for title, index in row['supporting_facts']:
            size = len(context.get(title, []))
            if title not in context or not 0 <= index < size:
                bad.append({
                    'title': title,
                    'index': index,
                    'sentence_count': size,
                })
        if bad:
            rejected.append({
                'id': row['_id'],
                'invalid_support': bad,
            })
        else:
            kept.append(row)

    assert kept, f'{source}: 没有有效样本'
    outputs = {
        source.with_suffix('.clean.json'): kept,
        source.with_suffix('.rejected.json'): rejected,
    }
    assert not any(p.exists() for p in outputs), '清洗输出已存在，请先检查'
    for path, content in outputs.items():
        with path.open('x', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    print(f'{name}: 原始 {len(rows)}，保留 {len(kept)}，剔除 {len(rejected)}')
    for item in rejected:
        print(item)
