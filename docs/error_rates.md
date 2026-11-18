# Error rates on declared bases

`misnomer.rates` computes character and word error rates (CER, WER) so that
the two are comparable with each other and across systems. It is model-free:
no language model, embedder or dictionary is involved.

## Why

A character rate computed on raw strings and a word rate computed on
normalised words are not on the same footing: the first charges a system for
case, punctuation and spacing, the second does not. On a reference whose
punctuation is stored as separate tokens (`It 's a pity , he said .`), a
system that writes ordinary spacing is then charged in characters for every
space it did not copy, and its WER can come out below its CER. Computing both
rates from one stated sequence removes that artefact. It does not make
WER ≥ CER a law: a long insertion inside one word costs many characters and
one word.

## Definition `error-rates/1`

1. Both strings are brought to Unicode NFC (canonical equivalence only; never
   NFKC, which would turn long s into s).
2. If a reference convention is given, it is applied to both strings.
3. Two bases are computed:

| Basis | Characters | Words |
|---|---|---|
| `verbatim` | the string with every run of whitespace (line breaks included) as one space, ends trimmed; nothing else altered | its space-separated tokens |
| `content` | the content tokens joined by single spaces | see below |

A `content` token is made from each whitespace-separated token in four steps:
typographic apostrophes (U+2019, U+02BC) are read as `'`; everything before
the first and after the last letter or number is removed, except combining
marks directly following that last letter or number; a token with no letter
or number is dropped; the rest is lower-cased with `str.lower` and brought
back to NFC. "Letter or number" means Unicode general categories L and N:
decimal digits, and also fractions, superscripts and enclosed numbers
(`½`, `²`, `①`).

So on the `content` basis: case and punctuation or symbols at token edges do
not count; a token made only of punctuation or symbols (`,`, `-`, `&`, `#`)
is not a word; characters inside a token are kept, so `well-known` is one
word and differs from `well known`, `don't` differs from `dont`, and a dash
inside a token stays what it is; an accent or abbreviation mark is never
dropped; long s and ß are not folded (`str.lower`, not casefold).

For each basis the result holds four integers: `char_edits`, `ref_chars`,
`word_edits`, `ref_words`. Edits are Levenshtein distances at unit cost,
without transpositions. A rate is edits over reference length (`None` when the
reference is empty); for a corpus, the **sum** of edits over the **sum** of
reference lengths (`pool()`), never a mean of per-line ratios.

```python
from misnomer import error_rates
from misnomer.rates import pool

a = error_rates("It's a pity, he said.", "It 's a pity , he said .",
                convention="tokenised-english")
b = error_rates("The horse stood.", "The house stood .", convention="tokenised-english")
a.content.wer            # 0.0
a.convention             # ConventionIdentity(name='tokenised-english', version='1.0', rules_sha256='…')
pool([a, b]).content.wer # 0.125
```

Every result records the definition, the Unicode form and the identity of the
convention (name, version, hash of its rules). `pool()` refuses to sum results
that differ in any of them. Inputs must be text or `None` (an absent answer,
scored as the empty string).

The functions do not clean the prediction. Whether a prediction was stripped
of chat boilerplate beforehand is the caller's decision and should be reported
with the figures.

### What holds the definition still

`tests/fixtures/error_rates_golden.json` pins the eight integers of 44 pairs
with and without the built-in convention; `tests/test_rates.py` and
`tests/test_conventions.py` pin the individual rules of the tokeniser and of
the convention, and the hash of the built-in rule list. A deliberate change is
released as a new definition (`error-rates/2`) or a new convention version,
with the fixtures regenerated.

## Reference conventions

A convention (`misnomer.conventions`) is a named, versioned, ordered list of
regular-expression rules describing how a reference text is laid out. It is
applied to both strings, never by default; an unknown name, an unknown key or
a malformed rule is an error. The rule list is applied repeatedly until the
text no longer changes, so applying a convention to its own output changes
nothing; a rule set that does not settle is an error. Patterns are compiled
without flags. Built-ins live in `src/misnomer/data/conventions.json`; a
custom one is passed inline as `{"name", "version", "description", "rules"}`.

`tokenised-english` (version 1.0) is for references in which punctuation
marks and clitics are separate tokens, as in the Brown, LOB and Penn Treebank
tradition. It changes whitespace only. In order:

1. every run of whitespace becomes one space;
2. double quotation marks (`"`, `“`, `”`) at the edge of a word are detached
   from it on both strings — the direction of a straight quotation mark cannot
   be decided within a line, so they are never re-attached;
3. the space before `, . ; : ! ?` is removed when the marks end a token, alone
   or followed only by closing brackets and further such marks;
4. the space before a closing bracket and after an opening bracket is removed;
5. the clitics `'s 'll 're 'd 'm 've` (any letter case, `'` or `’`) rejoin a
   preceding word;
6. the Penn-style split negation `n't` rejoins a preceding word.

Limits, by design:

- **Single quotation marks are left alone**, because a detached one cannot be
  told from an apostrophe, and detaching them is not neutral for every system.
  The `verbatim` basis therefore still charges a system that attaches single
  quotation marks where the reference detaches them. In the IAM evaluation
  this is about 0.3 to 0.8 percentage points of verbatim WER for the
  vision-language models and nothing for a recogniser trained on the
  reference's own layout. Typographic quotation marks are detached but not
  replaced by straight ones, so their glyphs still count on `verbatim`. The
  `content` basis is unaffected by either.
- A quoted word that equals a clitic (`he said 'Re: …'`) is joined to the
  preceding word. Spaced initialisms (`B B C`) are not joined.

## What the rates do not settle

- A hyphen with spaces around it (`well - known`) is a spacing difference in
  the output and costs word edits on either basis.
- Lower-casing is part of the `content` basis; a case-sensitive figure is the
  `verbatim` one.
- Markup left in a reference (annotation codes fused to words, a double quote
  standing for a diaeresis) is charged to any system that does not reproduce
  it; cleaning it is a corpus decision and belongs in the evaluation layer.
- The scorer's legacy `wer` and `cer` report fields are unchanged and are not
  on one basis: `wer` comes from the word alignment (case- and
  punctuation-insensitive), `cer` from the boilerplate-stripped prediction and
  the raw reference. Use `error_rates` for figures that are compared with each
  other. The report itself will carry these counts from scorer version 1.2.

## In the IAM evaluation

    uv run python evaluation/analysis.py rates --out evaluation/derived/rates_v1/error_rates.json

computes the rates from the **raw model outputs** (deduplicated, last write
wins), under the `tokenised-english` convention, without any boilerplate
stripping, and needs neither the auditor nor rescored files. It prints two
tables — each system on its own line set (all 10,373 lines; the PyLaia rows on
the test split, because that model was trained on the rest), and every system
on the test split, which is the like-for-like basis for any comparison with
PyLaia — with these views:

| View | Lines | A failed line |
|---|---|---|
| answered | not empty, not flagged by the refusal test, not over-long (`len > 2·len(reference) + 20`): the "Lines" column of the table | is left out |
| all lines | every reference line | is scored as fully deleted (an empty prediction), not on its own text, so one runaway output cannot outweigh a page |

A prediction flagged by the refusal test whose reference also reads like a
refusal (`I can't recall …`) is a transcription: it is outside the answered
lines, as in the published table, and is scored as given in the all-lines view.

The two tests are those of the published table and let some non-transcriptions
through: 40 Gemini lines hold the model's own reasoning cut off mid-way, and 4
GPT-4o lines are non-answers the refusal test does not catch. They are listed
by index in `META_OUTPUT` (established by reading) and reported as a
sensitivity: the answered view without them, and the all-lines view with them
as failures. They matter: Gemini's answered content CER is 1.44 with them and
0.97 without.

The GPT-5 row merges a second pass over the 2,264 lines its first pass
returned empty (larger token budget, same prompt and model id; the model
snapshot was recorded for neither pass).

The output file records the commit and whether the code differs from it,
hashes of the code that ran and of every input, package versions, the
definition, the convention and the interval settings. Its 95 % intervals are
percentile bootstraps over lines; lines of one form are written by one hand
and are not independent, so the intervals are somewhat too narrow, and they
are per system — they do not test a difference between two systems.
`tests/fixtures/iam_error_rates_pinned.json` pins every integer of both
tables; they were confirmed line by line by an independent implementation.
