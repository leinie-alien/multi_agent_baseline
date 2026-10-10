"""Read-only collaboration audit; run with .venv/bin/python scripts/analyze_collaboration.py.

Uses the experiment's extractor, equivalence checker and voting tie-break rule.
PASS/FAIL below mean control-flow acceptance/rejection, NOT words in verifier prose.
All rates include explicit denominators. --details prints auditable PEV transitions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import RouterConfig
from src.multi_agent.roles import _all_agree, _parse_category
from src.extract_answer import extract_answer, is_equiv, last_boxed_only_string
from src.methods.self_consistency import majority_vote

METHODS = ('direct', 'cot', 'self_consistency', 'debate', 'pev', 'router', 'mde')
SEEDS = (41, 42, 43)

# subject 仅用于离线评估,不进入 Router 推理输入。
SUBJECT_LOOKUP = {
    q['question_id']: q['subject']
    for q in json.loads((ROOT / 'data' / 'math500_200.json').read_text())
}
assert len(SUBJECT_LOOKUP) == 200

# Human reading of FINAL candidate verdicts in these frozen traces, not keyword
# counting (FAIL often occurs in quoted instructions or abandoned calculations).
# A rejected correct answer need not have a valid proof: this is answer-level
# rejection, NOT an assertion that every verifier critique is mathematically false.
AUDIT_DIGEST = 'c018541163c9ddf46ba7d5d05dcfae8fda46f46987551937fb4f04b8ad13973e'
CORRECT_REJECTED = {
    41: {13, 37, 49, 54, 58, 74, 114, 119, 124, 126, 152, 161, 169, 170, 179, 183, 194, 197},
    42: {19, 51, 58, 92, 93, 124, 130, 147, 163, 179, 185, 196},
    43: {17, 111, 130, 145, 147, 159, 176},
}
# Initial executor vote is wrong, verifier rejects that candidate and returns
# a scoring-correct alternative. q086/seed42 is only a rationalization change.
FAIL_RESCUED_VOTE = {
    41: {51, 108, 167, 188, 191},
    42: {86, 119, 169, 194, 197},
    43: {97, 120, 179, 183, 187, 194, 197},
}


def answer(t):
    return extract_answer(t['text'], allow_fallback=t['finish_reason'] != 'length')


def vote(ts):
    return majority_vote([answer(t) for t in ts])[0]


def rate(n, d):
    return f'{n}/{d} ({100*n/d:.2f}%)' if d else f'{n}/{d} (N/A)'


def transition(a, b):
    return ('right' if a else 'wrong') + '->' + ('right' if b else 'wrong')


def debate(rows):
    counts = Counter()
    changes = Counter()
    judge = Counter()
    levels = {i: Counter() for i in range(1, 6)}
    for r in rows:
        ts = r['agent_trace']
        sol = [t for t in ts if t['role'].startswith('solver_')]
        ref = [t for t in ts if t['role'].startswith('refine_')]
        jud = [t for t in ts if t['role'] == 'judge']
        assert len(sol) == 3
        initial = is_equiv(vote(sol), r['ground_truth'])
        counts[transition(initial, r['correct'])] += 1
        levels[r['level']][transition(initial, r['correct'])] += 1
        counts['initial_correct'] += initial
        if not ref:
            assert not jud and r['rounds_used'] == 1
            assert all(is_equiv(answer(t), answer(sol[0])) for t in sol)
            counts['early'] += 1
            counts['early_correct'] += r['correct']
        else:
            assert len(ref) == 3
            counts['refine_questions'] += 1
            for t in sol:
                u = next(u for u in ref if u['role'] == t['role'].replace('solver_', 'refine_'))
                changes[transition(is_equiv(answer(t), r['ground_truth']),
                                   is_equiv(answer(u), r['ground_truth']))] += 1
            if jud:
                assert len(jud) == 1
                before = is_equiv(vote(ref), r['ground_truth'])
                raw = answer(jud[0])
                final = raw if raw is not None else vote(ref)
                assert final == r['prediction']
                judge[transition(before, r['correct'])] += 1
                judge['vote_correct'] += before
                judge['judge_correct'] += r['correct']
                judge['fallback'] += raw is None
                judge['n'] += 1
    print('\nDEBATE rounds:', dict(sorted(Counter(r['rounds_used'] for r in rows).items())))
    print('question counts:', dict(counts))
    print('early stop:', rate(counts['early'], len(rows)))
    print('solver transitions (denominator = refined questions * 3):')
    for k, v in sorted(changes.items()):
        print(k, rate(v, sum(changes.values())))
    print('judge:', dict(judge))
    print('initial vote -> final by level:', {k: dict(v) for k, v in levels.items()})


def pev(rows, details=False):
    c = Counter()
    retries = Counter()
    transitions = Counter()
    levels = {i: Counter() for i in range(1, 6)}
    losses, fixes = [], []
    for r in rows:
        attempts, current = [], []
        for t in r['agent_trace']:
            if t['role'].startswith('executor_'):
                current.append(t)
            elif t['role'] == 'verifier':
                assert current
                attempts.append((current, t))
                current = []
        assert attempts and not current
        retries[len(attempts)-1] += 1
        first, _ = attempts[0]
        initial = is_equiv(vote(first), r['ground_truth'])
        good = [is_equiv(answer(t), r['ground_truth']) for t in first]
        c['initial_vote_correct'] += initial
        c['executor_0_correct'] += good[0]
        c['executor_1_correct'] += good[1] if len(good) > 1 else 0
        c['any_initial_correct'] += any(good)
        c['both_initial_correct'] += all(good)
        c['final_correct'] += r['correct']
        c['empty_final'] += r['prediction'] == ''
        c['empty_final_with_correct_candidate'] += r['prediction'] == '' and any(good)
        c['first_pass'] += answer(attempts[0][1]) is not None
        c['fallback'] += answer(attempts[-1][1]) is None
        c['any_correct_lost'] += any(good) and not r['correct']
        c['both_correct_lost'] += all(good) and not r['correct']
        c['none_correct_rescued'] += not any(good) and r['correct']
        transitions[transition(initial, r['correct'])] += 1
        levels[r['level']][transition(initial, r['correct'])] += 1
        for ex, v in attempts:
            accepted = answer(v) is not None
            egood = [is_equiv(answer(t), r['ground_truth']) for t in ex]
            c['verification_events'] += 1
            c['boxed_events'] += last_boxed_only_string(v['text']) is not None
            c['pass'] += accepted
            c['pass_wrong'] += accepted and not is_equiv(answer(v), r['ground_truth'])
            c['fail'] += not accepted
            c['fail_any_correct'] += not accepted and any(egood)
            c['fail_vote_wrong_rescued'] += (not accepted and
                not is_equiv(vote(ex), r['ground_truth']) and r['correct'])
            c['fail_all_wrong_rescued'] += not accepted and not any(egood) and r['correct']
        last_ex, last_v = attempts[-1]
        expected = answer(last_v) if answer(last_v) is not None else vote(last_ex)
        assert expected == r['prediction'], (r['seed'], r['question_id'])
        item = dict(seed=r['seed'], id=r['question_id'], level=r['level'],
                    gt=r['ground_truth'], candidates=[answer(t) for t in first],
                    final=r['prediction'], correct=r['correct'])
        if any(good) and not r['correct']:
            losses.append(item)
        if not initial and r['correct']:
            fixes.append(item)
    print('counts:', dict(c))
    print('retries:', {i: rate(retries[i], len(rows)) for i in range(3)})
    print('initial executor vote -> final:', dict(transitions))
    print('initial executor vote -> final by level:', {k: dict(v) for k, v in levels.items()})
    for key in ('first_pass', 'fallback', 'any_correct_lost', 'none_correct_rescued'):
        print(key, rate(c[key], len(rows)))
    print('PASS wrong:', rate(c['pass_wrong'], c['pass']))
    print('FAIL interception of correct candidate:', rate(c['fail_any_correct'], c['fail']))
    print('FAIL rescue:', rate(c['fail_vote_wrong_rescued'], c['fail']))
    if details:
        print('Correct candidate lost:', json.dumps(losses, ensure_ascii=False))
        print('Initial vote rescued:', json.dumps(fixes, ensure_ascii=False))


def router(rows, subject_lookup):
    categories = RouterConfig().categories
    groups = {True: [], False: []}
    retries = Counter()
    for r in rows:
        trace = r['agent_trace']
        routed = next(t for t in trace if t['role'] == 'router')
        predicted = _parse_category(routed['text'], categories)
        groups[predicted == subject_lookup[r['question_id']]].append(r)
        retries[sum(t['role'] == 'specialist' for t in trace) - 1] += 1
    print('routing accuracy:', rate(len(groups[True]), len(rows)))
    for matched, subset in groups.items():
        print('final accuracy with routing ' + ('correct:' if matched else 'incorrect:'),
              rate(sum(r['correct'] for r in subset), len(subset)))
    print('feedback retry rate:', rate(sum(n for k, n in retries.items() if k > 0), len(rows)))
    print('retries:', {k: rate(n, len(rows)) for k, n in sorted(retries.items())})
    print('| Subject | Accuracy | Mean total tokens |')
    print('|---|---:|---:|')
    for subject in categories:
        subset = [r for r in rows if subject_lookup[r['question_id']] == subject]
        accuracy = rate(sum(r['correct'] for r in subset), len(subset))
        tokens = f"{mean(r['total_tokens'] for r in subset):,.1f}" if subset else 'N/A'
        print(f'| {subject} | {accuracy} | {tokens} |')


def mde(rows):
    counts = Counter()
    changes = Counter()
    for r in rows:
        solvers = [t for t in r['agent_trace'] if t['role'].startswith('solver_')]
        reconciler = [t for t in r['agent_trace'] if t['role'] == 'reconciler']
        assert len(solvers) >= 2
        answers = [answer(t) for t in solvers]
        if _all_agree(answers):
            assert not reconciler and r['rounds_used'] == 1
            counts['early'] += 1
        else:
            assert len(reconciler) == 1 and r['rounds_used'] == 2
            counts['reconcile'] += 1
            initial, _ = majority_vote(answers)
            changes[transition(is_equiv(initial, r['ground_truth']), r['correct'])] += 1
            raw = answer(reconciler[0])
            assert r['prediction'] == (raw if raw is not None else initial)
            counts['fallback'] += raw is None
            # 文本相同只是答案来源的弱信号,不能据此证明是否独立推导。
            if raw is not None:
                counts['matches_solver' if raw in answers else 'different_from_all'] += 1
    print('solve all-agree / early stop:', rate(counts['early'], len(rows)))
    print('disagreement:', rate(counts['reconcile'], len(rows)))
    print('initial solver majority vote -> final (reconciled questions):')
    for before in (False, True):
        for after in (False, True):
            key = transition(before, after)
            print(key, rate(changes[key], counts['reconcile']))
    print('reconciler parse failure / solver vote fallback:', rate(counts['fallback'], counts['reconcile']))
    parsed = counts['reconcile'] - counts['fallback']
    print('reconciler answer matches a solver (text comparison):', rate(counts['matches_solver'], parsed))
    print('reconciler answer differs from all solvers (text comparison):', rate(counts['different_from_all'], parsed))


def semantic_audit(rows):
    digest = hashlib.sha256()
    for r in rows:
        digest.update(json.dumps([r['seed'], r['question_id'], r['ground_truth'], r['agent_trace']],
                                 sort_keys=True, ensure_ascii=False).encode())
    if digest.hexdigest() != AUDIT_DIGEST:
        print('Manual verdict audit skipped: trace fingerprint differs from reviewed dataset.')
        return
    print('\nMANUALLY REVIEWED FINAL VERDICTS (question-level, frozen trace fingerprint verified)')
    print('Answer-correct rejection can concern proof quality; not every such rejection causes a lost point.')
    for seed in (*SEEDS, 'all'):
        c = Counter()
        subset = [r for r in rows if seed == 'all' or r['seed'] == seed]
        for r in subset:
            q = int(r['question_id'][1:])
            rejected = q in CORRECT_REJECTED[r['seed']]
            rescued = q in FAIL_RESCUED_VOTE[r['seed']]
            c['correct_answer_rejected'] += rejected
            c['rejection_and_final_wrong'] += rejected and not r['correct']
            c['fail_rescued_initial_vote'] += rescued
            # All nonempty wrong final answers in this fixed dataset were
            # endorsed in the verifier's final decision (manually checked).
            c['endorsed_final_wrong'] += bool(r['prediction']) and not r['correct']
        print(seed, {k: rate(v, len(subset)) for k, v in c.items()})
    print('Correct-answer rejection question IDs:', CORRECT_REJECTED)
    print('FAIL rescue question IDs:', FAIL_RESCUED_VOTE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results_no_thinking')
    parser.add_argument('--details', action='store_true')
    args = parser.parse_args()
    runs = {}
    reference = None
    for method in METHODS:
        for seed in SEEDS:
            folder = args.results / method / f'run_seed{seed}'
            rows = [json.loads(s) for s in (folder / 'per_question.jsonl').read_text().splitlines() if s.strip()]
            summary = json.loads((folder / 'summary.json').read_text())
            assert len(rows) == summary['num_questions'] == 200
            assert len({r['question_id'] for r in rows}) == 200
            assert abs(mean(r['correct'] for r in rows) - summary['accuracy']) < 1e-10
            assert abs(mean(r['total_tokens'] for r in rows) - summary['avg_total_tokens']) < 1e-8
            signature = {r['question_id']: (r['level'], r['ground_truth']) for r in rows}
            if reference is None:
                reference = signature
            assert signature == reference
            for r in rows:
                assert r['correct'] == is_equiv(r['prediction'], r['ground_truth'])
                assert r['total_tokens'] == r['input_tokens'] + r['output_tokens']
                r['seed'] = seed
            runs[method, seed] = rows
    print(f'Validated {len(runs)} runs: same 200 questions, levels and ground truths; summaries and final scores match.')
    print('Correctness uses existing is_equiv (not a symbolic mathematical equivalence oracle).')
    print('PEV PASS = extracted verifier answer exists; FAIL = no extractable answer.')
    print('A candidate-level FAIL in prose is not a feedback retry.')
    for method in METHODS:
        accs = [100 * mean(r['correct'] for r in runs[method, s]) for s in SEEDS]
        print(method, 'accuracy by seed:', accs, 'sample SD (pp):', round(stdev(accs), 3))
    debate([r for s in SEEDS for r in runs['debate', s]])
    for seed in (*SEEDS, 'all'):
        print('\nPEV', seed)
        rows = [r for s in SEEDS if seed == 'all' or s == seed for r in runs['pev', s]]
        pev(rows, args.details and seed != 'all')
    print('\nROUTER')
    router([r for s in SEEDS for r in runs['router', s]], SUBJECT_LOOKUP)
    print('\nMDE')
    mde([r for s in SEEDS for r in runs['mde', s]])
    semantic_audit([r for s in SEEDS for r in runs['pev', s]])
    print('\nLEVEL sizes per seed:', dict(sorted(Counter(r['level'] for r in runs['direct', 41]).items())))
    for metric in ('correct', 'total_tokens'):
        print('\n' + metric + ' (mean of three seed means)')
        print('| Method | L1 | L2 | L3 | L4 | L5 |')
        print('|---|---:|---:|---:|---:|---:|')
        for method in METHODS:
            vals = [mean(mean(r[metric] for r in runs[method, s] if r['level'] == level)
                         for s in SEEDS) for level in range(1, 6)]
            cells = [f'{100*x:.2f}%' if metric == 'correct' else f'{x:,.1f}' for x in vals]
            print('| ' + ' | '.join([method, *cells]) + ' |')
    print('\nPaired comparison vs self_consistency (same seed and question):')
    for method in ('debate', 'pev', 'router', 'mde'):
        pairs = Counter()
        for seed in SEEDS:
            sc = {r['question_id']: r for r in runs['self_consistency', seed]}
            for r in runs[method, seed]:
                pairs[transition(sc[r['question_id']]['correct'], r['correct'])] += 1
        print(method, dict(pairs))


if __name__ == '__main__':
    main()
