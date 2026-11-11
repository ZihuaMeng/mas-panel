# qwen 3b on the nsmallest panel

I reran this four-agent case on a Colab T4 to check whether reading activations affects the answers and to see how much one review answer depends on the decoding seed. Capture on and off gave identical outputs on all 16 calls. With the prompt fixed, changing the seed changed some verdicts while the first-token logits and activation vector stayed identical.

## the case

`nsmallest(2, [6, 1, 4, 1, 1])` returns `[1, 1]` in CPython 3.12.0. The claim says it returns `[1, 4]`, so the correct verdict is **No**. I used panel seed 29 and gave each reviewer a different check. Both reviews reuse each agent’s check, initial answer, and decoding seed. The peer review adds the other three agents’ initial labels in the seed-29 order.

| Agent | Assigned check | Expected check output | Initial | Private | Peer |
| --- | --- | --- | --- | --- | --- |
| 0 | Boundary | `[1, 1, false]` | Yes | Yes | Yes |
| 1 | Invariant | `[2, 2]` | Yes | Yes | Yes |
| 2 | Simulation | `[1, 3]` | Unknown | No | No |
| 3 | Trace | `[6, 4]` | No | Yes | Yes |

The old and new runs gave the same verdicts. Agent 1 got the check right and answered Yes. Agent 0 hit the 48-token check limit with an unfinished array. Agents 2 and 3 gave check values that differed from the references. Every verdict was parseable. The initial vote was undecided. In both reviews, agent 3 switched to Yes and agent 2 supplied the only correct No. The group voted **Yes** three to one. [historical calls](historical.jsonl) · [new calls](colab_trace.jsonl)

## activation capture and decoding seeds

I loaded `Qwen/Qwen2.5-Coder-3B-Instruct` at `488639f1ff808d1d3d0ba301aef8c11461451ec5` in FP16. I ran the panel twice and saved first-token logits both times. The second run also read decoder block 17 (zero-based) at the last prompt token. All 16 matched requests had identical prompt tokens, generated tokens, text, parsed verdicts, and first-token logits. The requests and generated tokens also matched all 16 historical calls for this case. The new runs generated one request at a time, so the historical match is a check on this panel alone.

For agent 3, I replayed each complete historical review prompt with the original decoding seed, `429618993130734582`, and seven more seeds, `11, 21, 31, 41, 51, 61, 71`. Within each prompt, the token IDs, all 151,936 first-token logits, and all 2,048 block values matched across the eight draws. The maximum absolute difference was zero.

| Frozen prompt | Prompt tokens | Yes logit | No logit | Unknown logit | First-token No probability at T=0.3 | Generated answers |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Private | 714 | 19.5000 | 18.7656 | 18.4844 | 7.72% | 7 Yes, 1 No |
| Peer | 741 | 19.7656 | 19.2031 | 18.4688 | 13.15% | 6 Yes, 2 No |

Both reviews returned No at seed 61. At seed 41, private review returned Yes and peer review returned No. Both returned Yes at the other six seeds. The table gives first-token scores for the exact label spellings and probabilities normalized over the full vocabulary at temperature 0.3. Adding the peer labels narrows the Yes–No logit gap from 0.734375 to 0.5625 for this prompt. The comparison covers one selected panel, with no steering applied.

## rerun

Run from the repo root on a CUDA machine or in a Colab GPU runtime. I used Python 3.13.16, PyTorch `2.14.1+cu130`, and Transformers `5.22.1`. In Colab, keep the CUDA-enabled PyTorch installation.

```sh
python -m pip install 'transformers==5.22.1'
PYTHONPATH=src python scripts/run_nsmallest_capture.py --output runs/nsmallest_capture
```

Use a new output directory. The script downloads the pinned checkpoint unless it is cached. Add `--local-files-only` to require the cache. It writes traces and activations for `off`, `on`, `inspect_private`, and `inspect_peer`, plus manifests and `report.json`. My run took 66.68 s with model loading and used 5.95 GiB peak allocated GPU memory. The 48 calls used 33,376 prompt tokens and 246 generated tokens. Setup and weight download are outside the timing.

The report has all seeds, generated token IDs, scores, per-call comparisons, and tensor hashes. The audit checks all 48 call-to-activation matches and the numerical results. The rerun writes the full tensors locally. [report](colab_report.json) · [audit](colab_audit.json) · [source and license](README.md)
