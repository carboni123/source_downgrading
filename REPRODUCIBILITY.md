# Reproducibility Log

Verification date: 2026-05-13.

Environment observed during this pass:

- Windows PowerShell
- Python 3.13.12
- pytest 8.4.2
- MiKTeX pdfTeX 3.141592653-2.6-1.40.28

## Non-Live Test Suite

Command:

```powershell
python -m pytest tests -m "not live"
```

Observed output:

```text
collected 378 items / 4 deselected / 374 selected
================ 374 passed, 4 deselected in 94.14s (0:01:34) =================
```

## Focused Source-Downgrading Fixture Gate

Command:

```powershell
python -m pytest tests/architecture/test_laundering.py tests/trace_memory/test_fr4_add_derived_no_laundering.py tests/trace_memory/test_benchmark.py -m "not live"
```

Observed output:

```text
collected 45 items
============================= 45 passed in 0.55s ==============================
```

## Committed Result-File Verification

Live/API benchmarks were not rerun in this pass. They use paid model calls and
the task only required validating committed outputs unless credentials and cost
were clearly acceptable. The following command reads the committed JSON files
and recomputes headline rates.

Command:

```powershell
$culture = [System.Globalization.CultureInfo]::InvariantCulture
function F($n) { return ([double]$n).ToString('0.00', $culture) }
$multi = Get-Content results\architecture\laundering_validation_multiseed_summary.json -Raw | ConvertFrom-Json
$bench = Get-Content results\benchmarks\laundering_benchmark_results.json -Raw | ConvertFrom-Json
$product = Get-Content results\benchmarks\product_comparison_results.json -Raw | ConvertFrom-Json
$poison = Get-Content results\benchmarks\poisonedrag_results.json -Raw | ConvertFrom-Json
$sdm = $multi.policies.source_downgrading
"multiseed n=$($sdm.n_seeds) truth_mean=$(F $sdm.mean.derived_trust_ceiling_violation_rate) truth_std=$(F $sdm.std.derived_trust_ceiling_violation_rate) false_external_mean=$(F $sdm.mean.false_externalization_after_inference) provenance_mean=$(F $sdm.mean.provenance_chain_recall)"
foreach ($arm in @('no_source','provenance_only','trace_memory')) { $s = $bench.$arm; $o = $s.aggregate.overall; "$arm scenarios=$($s.n_scenarios) local=$(F $o.inference_laundering_rate.mean) truth=$(F $o.derived_trust_ceiling_violation_rate.mean) chain=$(F $o.chain_step_ceiling_violation_rate.mean) false_external=$(F $o.false_externalization_after_inference.mean) provenance=$(F $o.provenance_chain_recall.mean) cascade=$(F $o.cascade_invisibility_gap.mean)" }
foreach ($arm in @('vector','trace_memory','bash','bash_nolabels')) { $s = $product.aggregates.$arm; $strict = $s.n_correct / $s.n_questions; $def = $s.n_defensible_correct / $s.n_questions; $unsafe = $s.n_unsafe / $s.n_questions; $contam = $s.n_contam_unsafe / $s.n_contaminated; $clean = $s.n_clean_correct / $s.n_clean; $parse = $s.n_parse_error / $s.n_questions; "$arm n=$($s.n_questions) strict=$(F $strict) defensible=$(F $def) unsafe=$(F $unsafe) contam_unsafe=$(F $contam) clean_actionable=$(F $clean) parse_error=$(F $parse) api_calls=$($s.total_api_calls) tool_calls=$($s.total_tool_calls)" }
foreach ($arm in @('vector','vector_with_labels','trace_memory','bash','bash_nolabels')) { $s = $poison.aggregates.$arm; "$arm n=$($s.n) clean_acc=$(F ($s.n_correct / $s.n)) asr=$(F ($s.n_target / $s.n)) n_correct=$($s.n_correct) n_target=$($s.n_target)" }
$c = $poison.classifier.confusion_matrix.adversarial; $clean = $poison.classifier.confusion_matrix.clean; $low = $c.fabricated_or_uncertain + $c.simulation + $c.inference; "poison_classifier adversarial_low_trust=$low adversarial_external=$($c.external) clean_external=$($clean.external) clean_nonexternal=$($clean.fabricated_or_uncertain)"
```

Observed output:

```text
multiseed n=20 truth_mean=0.00 truth_std=0.00 false_external_mean=0.00 provenance_mean=1.00
no_source scenarios=163 local=0.72 truth=1.00 chain=1.00 false_external=0.07 provenance=0.00 cascade=0.28
provenance_only scenarios=163 local=0.00 truth=0.45 chain=0.43 false_external=0.07 provenance=1.00 cascade=0.45
trace_memory scenarios=163 local=0.00 truth=0.00 chain=0.00 false_external=0.00 provenance=1.00 cascade=0.00
vector n=139 strict=0.24 defensible=0.61 unsafe=0.29 contam_unsafe=0.36 clean_actionable=0.50 parse_error=0.00 api_calls=306 tool_calls=0
trace_memory n=139 strict=0.22 defensible=0.96 unsafe=0.00 contam_unsafe=0.00 clean_actionable=0.82 parse_error=0.00 api_calls=306 tool_calls=0
bash n=139 strict=0.76 defensible=0.81 unsafe=0.00 contam_unsafe=0.00 clean_actionable=0.07 parse_error=0.00 api_calls=1159 tool_calls=885
bash_nolabels n=139 strict=0.76 defensible=0.80 unsafe=0.00 contam_unsafe=0.00 clean_actionable=0.07 parse_error=0.01 api_calls=1270 tool_calls=1001
vector n=100 clean_acc=0.56 asr=0.45 n_correct=56 n_target=45
vector_with_labels n=100 clean_acc=0.82 asr=0.21 n_correct=82 n_target=21
trace_memory n=100 clean_acc=0.82 asr=0.22 n_correct=82 n_target=22
bash n=100 clean_acc=0.68 asr=0.27 n_correct=68 n_target=27
bash_nolabels n=100 clean_acc=0.65 asr=0.30 n_correct=65 n_target=30
poison_classifier adversarial_low_trust=356 adversarial_external=144 clean_external=196 clean_nonexternal=4
```

## Regenerating Non-Live Benchmark Artifacts

These commands are deterministic/non-live and can be used to regenerate the
corresponding artifact files.

```powershell
python examples/architecture/run_laundering_validation.py --output-dir results/architecture
python examples/architecture/run_source_inference_validation.py --output-dir results/architecture
python benchmarks/laundering_dataset.py --output benchmarks/data/laundering_dataset.jsonl
python benchmarks/run_laundering_benchmark.py --output-dir results/benchmarks
python benchmarks/source_boundary_dataset.py --output benchmarks/data/source_boundary_dataset.jsonl
python benchmarks/run_source_boundary_benchmark.py --output-dir results/benchmarks
python benchmarks/coupling_dataset.py --output benchmarks/data/coupling_dataset.jsonl
python benchmarks/run_coupling_benchmark.py --output-dir results/benchmarks
```

## Live/API Benchmark Reproduction

These commands require an `OPENAI_API_KEY` in `.env` or the environment and will
make paid API calls. They were not rerun in this verification pass.

```powershell
python benchmarks/run_product_comparison.py --full --dataset adversarial_reload_v2 --arms vector,trace_memory,bash,bash_nolabels --model gpt-4.1-mini --judge --out-dir results/benchmarks
python benchmarks/run_poisonedrag.py --data benchmarks/data/poisonedrag/nq.jsonl --arms vector,vector_with_labels,trace_memory,bash,bash_nolabels --model gpt-4.1-mini --classifier-model gpt-4.1-mini --limit 100 --seed 0 --out-dir results/benchmarks
```

## Paper Build

Command:

```powershell
Push-Location paper
& "C:\Users\DiegoPC\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode Source_Downgrading.tex
& "C:\Users\DiegoPC\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode Source_Downgrading.tex
Pop-Location
```

Observed output:

```text
Output written on Source_Downgrading.pdf (25 pages, 388081 bytes).
Transcript written on Source_Downgrading.log.
```

Log scan:

```powershell
rg -n "LaTeX Warning: Citation|LaTeX Warning: Reference|undefined references|Overfull|Underfull|Warning|Error|Fatal" paper\Source_Downgrading.log
```

Observed output:

```text
1035:Underfull \hbox (badness 1342) in paragraph at lines 257--258
```

No undefined references, undefined citations, fatal errors, or overfull boxes
were present in the final log.
