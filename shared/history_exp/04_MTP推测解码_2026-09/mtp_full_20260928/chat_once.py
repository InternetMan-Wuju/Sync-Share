#!/usr/bin/env python3
"""Send one text prompt to the single-host service and print the streamed reply."""
import argparse
import datetime
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt', nargs='?', help='Without this argument, enter a prompt interactively.')
    parser.add_argument('--url', default='http://10.120.72.23:30080')
    parser.add_argument('--model', default='glm-5.2')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--out-root', type=Path, default=BASE / 'chat_checks')
    args = parser.parse_args()
    prompt = args.prompt if args.prompt is not None else input('请输入提示词：')
    if not prompt.strip():
        parser.error('提示词不能为空')
    if args.max_tokens < 1:
        parser.error('--max-tokens 必须大于 0')

    out_dir = args.out_root / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir.mkdir(parents=True)
    payload = {
        'model': args.model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0,
        'max_tokens': args.max_tokens,
        'stream': True,
        'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': False},
    }
    write_json(out_dir / 'request.json', {'url': args.url.rstrip('/') + '/v1/chat/completions', 'body': payload})
    req = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    answer, reasoning, usage, finish_reason = [], [], None, None
    error, status, first_content_s = None, None, None
    start = time.perf_counter()
    print(f'服务：{args.url}  模型：{args.model}\n回复：', flush=True)
    try:
        with urllib.request.urlopen(req, timeout=300) as response, (out_dir / 'events.jsonl').open('w', encoding='utf-8') as events:
            status = response.status
            for raw_line in response:
                if not raw_line.startswith(b'data:'):
                    continue
                raw = raw_line[5:].strip()
                if raw == b'[DONE]':
                    break
                obj = json.loads(raw)
                elapsed = time.perf_counter() - start
                events.write(json.dumps({'elapsed_s': elapsed, 'response': obj}, ensure_ascii=False) + '\n')
                events.flush()
                if obj.get('error'):
                    raise RuntimeError(json.dumps(obj['error'], ensure_ascii=False))
                if obj.get('usage'):
                    usage = obj['usage']
                for choice in obj.get('choices', []):
                    if choice.get('finish_reason') is not None:
                        finish_reason = choice['finish_reason']
                    delta = choice.get('delta') or {}
                    text = delta.get('content') or ''
                    if text:
                        if first_content_s is None:
                            first_content_s = elapsed
                        answer.append(text)
                        print(text, end='', flush=True)
                    if delta.get('reasoning_content'):
                        reasoning.append(delta['reasoning_content'])
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode('utf-8', 'replace')
    except Exception as exc:
        error = str(exc)
    elapsed = time.perf_counter() - start
    text = ''.join(answer)
    (out_dir / 'reply.txt').write_text(text + '\n', encoding='utf-8')
    if reasoning:
        (out_dir / 'reasoning.txt').write_text(''.join(reasoning) + '\n', encoding='utf-8')
    result = {'http_status': status, 'finish_reason': finish_reason, 'usage': usage,
              'first_content_s': first_content_s, 'e2e_s': elapsed, 'error': error,
              'nonempty_reply': bool(text.strip()), 'semantic_check': 'pending_manual_review',
              'purpose': '人工检查自然语言回复；不计入性能实验'}
    write_json(out_dir / 'summary.json', result)
    print(f'\n\n结束原因：{finish_reason}；耗时：{elapsed:.2f} 秒\n记录：{out_dir}', flush=True)
    if finish_reason == 'length':
        print('达到输出上限，回复可能未结束；可调大 --max-tokens。', file=sys.stderr)
    if error or not text.strip():
        print('请求失败或没有正文：' + (error or '请查看保存的响应'), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
