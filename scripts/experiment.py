"""Configuration-driven workflow. Run through experiment.sh from the repository root."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request


def run(*args, env=None):
    subprocess.run([str(a) for a in args], check=True, env=env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["download", "config", "serve", "health", "eval", "smoke", "train", "merge", "report"])
    p.add_argument("--role", choices=["reader", "policy", "memory"], default="reader")
    p.add_argument("--setting", choices=["before", "summary", "after"], default="before")
    p.add_argument("--data")
    p.add_argument("--out")
    p.add_argument("--adapter")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--trace-rollouts", action="store_true")
    a = p.parse_args()
    e = os.environ
    reader_output = e.get("READER_OUTPUT", "text")
    if reader_output not in {"text", "json_schema"}:
        p.error("READER_OUTPUT must be text or json_schema")
    memory_output = e.get("MEMORY_OUTPUT", "text")
    if memory_output not in {"text", "json_schema"}:
        p.error("MEMORY_OUTPUT must be text or json_schema")
    tag = e["EXPERIMENT_ID"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        p.error("EXPERIMENT_ID must be a simple directory name")
    cfgdir = Path("configs/generated") / tag
    out = Path(a.out or (f"outputs/{tag}/eval-{a.setting}" if a.command == "eval" else f"outputs/{tag}/{a.command}"))
    uv = ["uv", "run", "--locked", "--extra", "train"]
    if memory_output == "json_schema":
        uv += ["--extra", "structured"]
    role = a.role.upper()
    port = e.get(f"{role}_PORT", "")
    alias = f"memsearch-{a.role}"
    if a.command in ('eval', 'smoke', 'train', 'merge'):
        snapshot = json.loads((cfgdir/'models.json').read_text())
        changed = [k for k,v in snapshot.items() if k not in {'READER_OUTPUT', 'MEMORY_OUTPUT'} and e.get(k) != v]
        if snapshot.get('READER_OUTPUT', 'text') != reader_output:
            changed.append('READER_OUTPUT')
        if snapshot.get('MEMORY_OUTPUT', 'text') != memory_output:
            changed.append('MEMORY_OUTPUT')
        if changed:
            p.error(f"Settings changed since config generation: {changed}; use a new EXPERIMENT_ID and run config")
    if a.command in ('smoke', 'train'):
        if Path(e['POLICY_MODEL']).resolve() != Path(e['READER_MODEL']).resolve():
            p.error('This simplified before/after workflow requires the same policy and reader base model; configure matching models or use the explicit per-role API workflow')
        if a.command == 'train' and len(e['TRAIN_GPUS'].split(',')) != int(e['TRAIN_PROCESSES']):
            p.error('TRAIN_PROCESSES must match the number of TRAIN_GPUS')
    if a.command == "download":
        if role == "MEMORY":
            p.error("memory is a local merged checkpoint; use merge")
        run(*uv, "hf", "download", e[f"{role}_REPO"], "--local-dir", e[f"{role}_MODEL"])
    elif a.command == "config":
        # Build TOML from typed values; never replace model strings in a template.
        base = {"experiment": {"name": tag, "training_status": "untrained-prompt-baselines"},
                "backend": {"kind": "api", "base_url": f"http://127.0.0.1:{e['READER_PORT']}/v1",
                            "model": "memsearch-reader", "api_key_env": "MEMSEARCH_API_KEY",
                            "temperature": 0.0, "max_tokens": int(e['MAX_NEW']), "timeout": 300, "reader_output": reader_output, "memory_output": memory_output},
                "budget": {"tokenizer": e['BUDGET_TOKENIZER']},
                "agent": {"memory": "decision", "policy": "fixed", "memory_budget": 512,
                          "evidence_budget": 1536, "max_searches": 3, "top_k": 2, "retrieval_scope": "candidate"}}
        cfgdir.mkdir(parents=True, exist_ok=False)
        for setting in ("before", "summary", "after"):
            cfg = json.loads(json.dumps(base))
            if setting == "summary":
                cfg['agent']['memory'] = 'summary'
            if setting == "after":
                cfg['experiment']['training_status'] = 'rl-checkpoint-evaluation'
                cfg['memory_backend'] = {"model": "memsearch-memory", "base_url": f"http://127.0.0.1:{e['MEMORY_PORT']}/v1"}
            content = ''.join(f"[{section}]\n" + ''.join(f"{k} = {json.dumps(v)}\n" for k,v in values.items()) + "\n" for section,values in cfg.items())
            (cfgdir / f"{setting}.toml").write_text(content)
        # Local provenance snapshot: excluded from Git along with generated configurations.
        keys = ['READER_REPO', 'READER_MODEL', 'POLICY_REPO', 'POLICY_MODEL', 'BUDGET_TOKENIZER', 'MAX_NEW', 'READER_PORT', 'MEMORY_PORT']
        (cfgdir / 'models.json').write_text(json.dumps({**{k:e[k] for k in keys}, 'READER_OUTPUT': reader_output, 'MEMORY_OUTPUT': memory_output}, indent=2))
        print(cfgdir)
    elif a.command in ('serve', 'health'):
        if role == 'POLICY':
            p.error('Use reader for baseline inference; policy is loaded locally by training')
        if a.command == 'serve':
            run('ninja', '--version')
            model = str(Path(e[f'{role}_MODEL']).expanduser().resolve())
            env = dict(e, CUDA_VISIBLE_DEVICES=e[f'{role}_GPU'])
            args = [str(Path(e['SERVING_ENV'])/'bin/vllm'), 'serve', model,
                    '--served-model-name', alias, '--host', '127.0.0.1', '--port', port,
                    '--dtype', 'bfloat16', '--max-model-len', e['MAX_MODEL_LEN'],
                    '--gpu-memory-utilization', e['GPU_MEMORY_UTILIZATION'], '--max-num-seqs', e['MAX_NUM_SEQS']]
            os.execvpe(args[0], args, env)
        else:
            url = f'http://127.0.0.1:{port}/v1'
            with urllib.request.urlopen(url+'/models', timeout=30) as response:
                models = json.load(response)
            if alias not in [m['id'] for m in models['data']]:
                raise ValueError(f'Expected model alias {alias}, got {models}')
            body = json.dumps({'model':alias, 'messages':[{'role':'user','content':'Reply with OK.'}], 'max_tokens':16, 'temperature':0}).encode()
            request = urllib.request.Request(url+'/chat/completions', data=body, headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=120) as response:
                print(response.read().decode())
    elif a.command == 'eval':
        run(*uv, 'memsearch', 'eval', '--config', cfgdir/f'{a.setting}.toml', '--data', a.data or e['DEV_DATA'], '--out', out)
    elif a.command in ('smoke', 'train'):
        env = dict(e, CUDA_VISIBLE_DEVICES=e['TRAIN_GPUS'].split(',')[0] if a.command == 'smoke' else e['TRAIN_GPUS'], ACCELERATE_MIXED_PRECISION='bf16')
        prefix = uv + (['memsearch-train'] if a.command == 'smoke' else ['accelerate', 'launch', '--config_file', 'configs/accelerate_8x3090.yaml', '--num_processes', e['TRAIN_PROCESSES'], '-m', 'memsearch.training'])
        steps = 2 if a.command == 'smoke' else a.steps
        args = ['grpo', '--role', 'memory', '--memory-mode', 'decision', '--model', e['POLICY_MODEL'], '--config', str(cfgdir/'before.toml'), '--data', e['TRAIN_DATA'], '--out', str(out), '--steps', str(steps), '--group-size', '2' if a.command == 'smoke' else '4', '--max-prompt', '4096', '--max-new', e['MAX_NEW'], '--save-every', str(steps), '--seed', '42']
        if memory_output == 'json_schema':
            args += ['--structured-memory']
        if a.trace_rollouts:
            args += ['--trace-rollouts']
        if a.adapter:
            args += ['--adapter', a.adapter]
        run(*prefix, *args, env=env)
    elif a.command == 'merge':
        if not a.adapter:
            p.error('merge requires --adapter')
        run(*uv, 'python', 'scripts/merge_adapter.py', '--model', e['POLICY_MODEL'], '--adapter', a.adapter, '--out', e['MEMORY_MODEL'])
    elif a.command == 'report':
        if not a.out:
            p.error('report requires --out pointing to an existing result directory')
        if (out/'summary.json').exists():
            print((out/'summary.json').read_text())
        from collections import Counter
        errors = Counter()
        roles = Counter()
        examples = []
        if (out/'trajectories.jsonl').exists():
            for line in (out/'trajectories.jsonl').read_text().splitlines():
                row = json.loads(line)
                errors.update(row['errors'])
                if row['errors'] and row['calls']:
                    call = row['calls'][-1]
                    roles.update([call['role']])
                    if len(examples) < 3:
                        examples.append({'task_id': row['task_id'], 'errors': row['errors'],
                                         **{k: call.get(k) for k in ('role', 'response', 'finish_reason', 'decoding_mode')}})
            print('Errors:', errors.most_common(10))
            print('Failure roles:', dict(roles))
            print(json.dumps(examples, ensure_ascii=False, indent=2))
        for path in sorted(out.glob('rank-*.jsonl')):
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            print(path.name, 'steps', len(rows), 'updated', sum(r['updated'] for r in rows), 'zero_variance', sum(r['zero_variance'] for r in rows))


if __name__ == '__main__':
    main()
