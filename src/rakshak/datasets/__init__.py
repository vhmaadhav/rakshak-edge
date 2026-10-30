"""Dataset pipeline for the scam-SMS and URL corpora.

Modules:
    common      constants, stable ids, dedup keys
    lang        deterministic en / hinglish / hi_deva / other detector
    template    template keys (URLs, numbers, amounts and phones masked)
    loaders     per-source readers that return a standard raw frame
    dedup       exact dedup, MinHash LSH near-duplicates, union-find campaigns
    sms         SMS corpus assembly (raw frame -> analysis-ready frame)
    urls        URL corpus assembly (normalisation, dedup, url_features)
    splits      campaign, temporal, source-holdout, lang-shift and legacy partitions
    confound    source-identity classifier and label-by-source baseline
    build       command-line entry point (python -m rakshak.datasets.build)

Build outputs go to data/processed/ (git-ignored). See docs/DATACARD.md.
"""
