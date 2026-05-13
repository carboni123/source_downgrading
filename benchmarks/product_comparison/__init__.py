"""Product comparison benchmark.

Multi-turn sessions across three arms:

* ``bash``         -- flat markdown files + LLM tool-use (glob/grep/read_file)
* ``vector``       -- in-memory cosine top-k + LLM (no source labels)
* ``trace_memory`` -- closed-loop MemoryAgent with reactivation envelope,
                      source-downgrading inscription on every LLM-derived
                      answer, and routing kwargs threaded from a small
                      regex post-classifier on the LLM's output.

The product question this benchmark settles is whether trace-memory's
closed-loop discipline matches bash-tool-use answer quality at meaningfully
lower token cost, while keeping safety violations strictly lower than both
baselines on adversarially-contaminated multi-turn sessions.
"""
