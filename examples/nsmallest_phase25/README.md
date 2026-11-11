# Phase 25 nsmallest case

`dataset.json` contains the original `real-code-cpython-nsmallest-2` item. The displayed function is from [CPython 3.12.0, Lib/heapq.py lines 463–521](https://github.com/python/cpython/blob/0fb18b02c8ad56299d6a2910be0bab8ad601ef24/Lib/heapq.py#L463-L521). The experiment prepended `from _heapq import _heapify_max, _heapreplace_max` and authored the input, claim, and checking questions. The CPython source is covered by the [included license](CPYTHON_LICENSE.txt).

`historical.jsonl` preserves the 16 original requests and outputs for the four different checking questions at panel seed 29. It comes from the corrected Phase 25 3B ledger at lines 157–160, 221–224, 285–288, and 349–352. The checkpoint was `Qwen/Qwen2.5-Coder-3B-Instruct` at `488639f1ff808d1d3d0ba301aef8c11461451ec5`, run in FP16 on a Colab T4.

| File | SHA-256 |
| --- | --- |
| Upstream `Lib/heapq.py` | `6d43277e5c76fc0f073cd388fcff852d14d068f6bb6d4886c340f8b75a1229a9` |
| `dataset.json` | `f4058eca13c759fa8c094e01113781ad31c9d0d7eccec5e81816a84872f54154` |
| `historical.jsonl` | `560ac79bd379d0fd56c0e20f411ddb0942bebc2ee3b3b68712c2a2b035a4cad0` |
| Corrected Phase 25 3B source ledger | `8ebfc3763e5969b951fb88672f066c2731316dcd0e3cc445d1d1ed18072d518a` |
| `CPYTHON_LICENSE.txt` | `3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf` |
