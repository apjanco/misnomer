from __future__ import annotations

import csv
import json
from pathlib import Path

import typer

from misnomer.config import ScorerConfig
from misnomer.highlight import highlight
from misnomer.report import SemanticErrorReport
from misnomer.scorer import score, score_batch

app = typer.Typer(help="Semantic error measurement for OCR/HTR output")


@app.command("score")
def score_command(
    predicted: Path = typer.Option(..., exists=True, dir_okay=False),
    ground_truth: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    scorer_version: str = typer.Option("1.0"),
    multilingual: bool = typer.Option(False),
    allow_download: bool = typer.Option(False),
    model_revision: str | None = typer.Option(None),
) -> None:
    pred_text = predicted.read_text(encoding="utf-8")
    gt_text = ground_truth.read_text(encoding="utf-8")

    cfg = ScorerConfig(
        scorer_version=scorer_version,
        use_multilingual_embedder=multilingual,
        allow_download=allow_download,
        local_files_only=not allow_download,
        model_revision=model_revision,
    )
    report = score(predicted=pred_text, ground_truth=gt_text, config=cfg)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")


@app.command("score-batch")
def score_batch_command(
    input: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    scorer_version: str = typer.Option("1.0"),
    multilingual: bool = typer.Option(False),
    allow_download: bool = typer.Option(False),
    model_revision: str | None = typer.Option(None),
) -> None:
    pairs: list[tuple[str, str]] = []
    with input.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pred = row.get("predicted", "")
            gt = row.get("ground_truth", "")
            pairs.append((pred, gt))

    cfg = ScorerConfig(
        scorer_version=scorer_version,
        use_multilingual_embedder=multilingual,
        allow_download=allow_download,
        local_files_only=not allow_download,
        model_revision=model_revision,
    )
    reports = score_batch(pairs=pairs, config=cfg)
    payload = [r.model_dump(mode="json") for r in reports]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.command("highlight")
def highlight_command(
    report: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., dir_okay=False),
    threshold: float = typer.Option(0.3, min=0.0, max=1.0),
) -> None:
    raw = report.read_text(encoding="utf-8")
    report_obj = SemanticErrorReport.model_validate_json(raw)
    html = highlight(report_obj, threshold=threshold)
    output.write_text(html, encoding="utf-8")


@app.command("version")
def version_command() -> None:
    typer.echo("misnomer package version 0.1.0")
    typer.echo("default scorer version 1.0")

