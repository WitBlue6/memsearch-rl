"""Compare matched stage-one evaluations; never infer convergence from a short run."""
import argparse
import json
from pathlib import Path


def compare(paths):
    loaded = {}
    for name, root in paths.items():
        root = Path(root)
        loaded[name] = {kind: json.loads((root/f'{kind}.json').read_text())
                        for kind in ('summary', 'manifest', 'config')}
    if len({v['manifest']['dataset_sha256'] for v in loaded.values()}) != 1:
        raise ValueError('Dataset fingerprints differ; use the same evaluation dataset')
    if len({v['summary']['examples'] for v in loaded.values()}) != 1:
        raise ValueError('Example counts differ')
    first = loaded['before']['config']
    for name, values in loaded.items():
        cfg = values['config']
        if cfg.get('budget') != first.get('budget'):
            raise ValueError(f'{name}: budget tokenizer differs')
        reader = {**cfg['backend'], **cfg.get('reader_backend', {})}
        reference_reader = {**first['backend'], **first.get('reader_backend', {})}
        if reader != reference_reader:
            raise ValueError(f'{name}: reader or decoding configuration differs')
        agent = {k:v for k,v in cfg['agent'].items() if k != 'memory'}
        if agent != {k:v for k,v in first['agent'].items() if k != 'memory'} or agent.get('policy') != 'fixed':
            raise ValueError(f'{name}: retrieval/budget settings differ or search is not fixed')
    if loaded['before']['config']['agent']['memory'] != loaded['after']['config']['agent']['memory']:
        raise ValueError('Before/after memory protocols differ')
    keys = ('em','f1','invalid','memory_support_recall','citation_support_recall','searches')
    rows = {name: {k: v['summary'][k] for k in keys} for name,v in loaded.items()}
    return {'examples': loaded['before']['summary']['examples'],
            'dataset_sha256': loaded['before']['manifest']['dataset_sha256'],
            'metrics': rows,
            'after_minus_before_f1': rows['after']['f1']-rows['before']['f1'],
            'after_minus_summary_f1': rows['after']['f1']-rows['summary']['f1'],
            'total_usage': {name:v['summary']['total_usage'] for name,v in loaded.items()},
            'note': 'Matched configuration checks passed; checkpoint identity and factuality require separate audit. A short run does not establish algorithmic improvement.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('before','summary','after','out'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    result = compare({k:getattr(args,k) for k in ('before','summary','after')})
    with Path(args.out).open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
