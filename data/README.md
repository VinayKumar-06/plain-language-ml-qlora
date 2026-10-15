# Dataset notes

`raw/examples.jsonl` contains 52 project-specific examples covering 13 concepts in foundations, data practice, and evaluation. Each concept has four instruction phrasings with one matching answer. Concept groups are split as a unit to prevent paraphrases about the same concept from crossing train, validation, and test.

The examples were prepared for this project and do not come from a third-party corpus. They have not received independent subject-matter review; check each answer before using the model to teach. The project MIT license applies to this original seed material. If you replace or augment it, record source, license, attribution, and any usage conditions here before training or publishing.

Prepare splits with:

```bash
make prepare
```

The resulting JSONL splits and manifest live in `data/processed/`, which is ignored by Git because they are reproducible from the raw source file.
