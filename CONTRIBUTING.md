# Contributing

Thank you for helping. Issues, questions and pull requests are welcome.

## Getting set up

```bash
git clone https://github.com/miranpoor/glossdex && cd glossdex
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The tests need no models: they use exact synthetic scores and a hashing embedder.

## What helps most

- **Adapters**: content sources (mail, notes, a document-management system), describers (other
  local runtimes such as llama.cpp, LM Studio or vLLM) and embedders.
- **Bug reports with `glossdex explain` output.** It shows every score and every decision, so a
  "this didn't find X" report is usually solvable from it alone.
- **Synonym classes** for vocabularies glossdex doesn't cover yet. Two words belong in a class
  only if a describing model might write one where a person would type the other, about the
  same thing. A loose class creates false full matches, so please explain each addition.

## Ground rules

- Changes to the result-set stages or the profile thresholds need a measurement: before and
  after on a labeled query set that includes queries with no answer. Report recall, wrong
  results per query and the no-answer rate, not just one aggregate.
- Keep the core dependency-free beyond NumPy and Pillow.
- By contributing you agree that your contribution is licensed under the AGPL-3.0 and that the
  project may also offer it under the commercial license. A short contributor license agreement
  will be added before outside contributions are merged.

Please follow the [code of conduct](CODE_OF_CONDUCT.md).
