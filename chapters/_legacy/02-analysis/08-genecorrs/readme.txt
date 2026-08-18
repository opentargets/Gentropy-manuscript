Initial filters(_load_and_filter_studies): GWAS only, has sumstats, no disqualifying QC flags, LDSC-compatible analysis flags, neff ≥ 10,000, ancestry must be determinable
Heritability filters(_merge_and_apply_h2_filters): inner join with successful h² runs, then h² > 0, h² < 1, h²/SE ≥ 2 (z² ≥ 4), lambda_gc between 0.8–2.5
Representative selection(_select_representatives): explode bytraitFromSourceMappedIds(disease ID), then pick thehighest neff study per (diseaseId, ld_ancestry) cell, with h²/SE as tiebreaker
Pair generation(_generate_pairs): all-vs-all combinations within each ancestry
