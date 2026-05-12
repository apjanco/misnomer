import asyncio
import json
import logging
import math
import os
from pathlib import Path

import pandas as pd
import srsly
from portkey_ai import AsyncPortkey
from tqdm.asyncio import tqdm

log = logging.getLogger(__name__)

RESULTS = Path(__file__).parents[2] / "evaluation" / "full_eval_results.jsonl"
OUT = Path(__file__).parents[2] / "evaluation" / "false_negatives_analysis.jsonl"
MODEL = "gpt-4-turbo"
CONCURRENCY = 20       # max simultaneous API calls
MAX_RETRIES = 5        # per-request retry attempts
BASE_BACKOFF = 2.0     # seconds; doubles each retry

SYSTEM_PROMPT = (
    "You will be given two texts, one is the ground truth and the other is the predicted text. "
    "You will be asked to compare the predicted text against the ground truth. "
    "Does the predicted text contain any complete words that differ from the ground truth? "
    "Ignore character-level differences, spacing and punctuation. "
    "Return only 'Yes' or 'No' and a brief explanation if the answer is 'Yes'."
)
USER_PROMPT = "GROUND TRUTH: {ground_truth}\nPREDICTED TEXT: {predicted_text}"


def _sanitize(d: dict) -> dict:
    """Replace NaN/Inf float values with None so json.dumps produces valid JSON."""
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in d.items()
    }


# Errors that are worth retrying (transient network / gateway / rate-limit)
try:
    import openai
    _RETRYABLE = (
        openai.InternalServerError,
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
    )
except ImportError:
    _RETRYABLE = (Exception,)


async def analyse_candidate(
    client: AsyncPortkey,
    sem: asyncio.Semaphore,
    candidate: dict,
    out_fh,
    pbar: tqdm,
) -> dict | None:
    """Call the API with exponential-backoff retries; append result immediately."""
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT.format(
                            ground_truth=candidate["ground_truth"],
                            predicted_text=candidate["predicted_text"],
                        )},
                    ],
                )
                result = _sanitize({**candidate, "analysis": response.choices[0].message.content})
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_fh.flush()
                pbar.update(1)
                return result

            except _RETRYABLE as exc:
                if attempt == MAX_RETRIES - 1:
                    log.error("Giving up on candidate %s after %d attempts: %s",
                              candidate.get("image_path"), MAX_RETRIES, exc)
                    pbar.update(1)
                    return None
                wait = BASE_BACKOFF * (2 ** attempt)
                log.warning("Attempt %d failed (%s: %s) — retrying in %.1fs",
                            attempt + 1, type(exc).__name__, exc, wait)
                await asyncio.sleep(wait)


async def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise SystemExit("Set the PORTKEY_API_KEY environment variable before running.")

    client = AsyncPortkey(api_key=api_key)

    data = srsly.read_jsonl(RESULTS)
    df = pd.DataFrame(data)
    df = df[df["model"] != "full_eval_results"]

    candidates = [
        _sanitize(r)
        for r in df[
            (df["semantic_error_count"] == 0)
            & (df["predicted_text"].fillna("").str.strip() != df["ground_truth"].fillna("").str.strip())
        ].to_dict(orient="records")
    ]

    # Resume: skip candidates already written to the output file
    done: set[tuple] = set()
    if OUT.exists():
        for r in srsly.read_jsonl(OUT, skip=True):
            done.add((r.get("model"), r.get("image_path")))
        print(f"Resuming — {len(done):,} already done, skipping")

    pending = [c for c in candidates if (c.get("model"), c.get("image_path")) not in done]
    print(f"{len(pending):,} candidates remaining (of {len(candidates):,} total)")

    sem = asyncio.Semaphore(CONCURRENCY)

    with OUT.open("a", encoding="utf-8") as out_fh:
        with tqdm(total=len(pending), desc="Analysing false negatives") as pbar:
            tasks = [analyse_candidate(client, sem, c, out_fh, pbar) for c in pending]
            await asyncio.gather(*tasks)

    total_written = sum(1 for _ in srsly.read_jsonl(OUT, skip=True))
    print(f"Done. {total_written:,} results in {OUT}")


if __name__ == "__main__":
    asyncio.run(main())

