"""End-to-end pipeline: acquire -> clean -> validate -> train -> score -> persist.

Run with `sentiment-pipeline` (installed script) or `python -m sentiment.pipeline`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import cleaning, config, data, model, validation

log = logging.getLogger("sentiment.pipeline")


def run(csv: str | None = None, force_download: bool = False, allow_synthetic: bool = True) -> dict:
    df = data.load_user_csv(csv) if csv else data.load_raw(force_download, allow_synthetic)
    log.info("Loaded %d raw rows", len(df))

    df = cleaning.clean_frame(df)
    df, report = validation.validate_frame(df)
    log.info("Validation kept %d/%d rows; dropped %s", report.rows_out, report.rows_in, report.dropped)
    for w in report.warnings:
        log.warning(w)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_CSV, index=False)
    config.QUALITY_JSON.write_text(json.dumps(report.to_dict(), indent=2))

    pipe, metrics = model.train_and_evaluate(df)
    log.info("Test F1 %.3f | ROC-AUC %.3f | CV F1 %.3f +/- %.3f",
             metrics.f1, metrics.roc_auc, metrics.cv_f1_mean, metrics.cv_f1_std)
    model.save(pipe, metrics)

    scored = model.score_frame(pipe, df)
    scored.to_csv(config.SCORED_CSV, index=False)
    log.info("Wrote %s", config.SCORED_CSV)
    return {"rows": len(scored), "metrics": metrics.to_dict(), "quality": report.to_dict()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Use your own CSV (columns: text, label[, source, review_date])")
    parser.add_argument("--force-download", action="store_true", help="Re-download the UCI data")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Fail instead of falling back to synthetic data when offline")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    result = run(args.csv, args.force_download, not args.no_synthetic)
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
