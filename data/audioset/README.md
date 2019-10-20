# Audioset Data

This directory contains labeled links to YouTube videos containing sounds of the specified type.
Files are layed out as follows:

- `ontology.json` AudioSet's dictionary mapping AudioSet human-readable feature names to feature IDs
- `eval_segments.csv`, `balanced_train_segments.csv`, `unbalanced_train_segments.csv` AudioSet's labeled YouTube links (labeled by feature ID)
- `ngmap.json` Our dictionary mapping our custom feature names to AudioSet's feature IDs (not yet populated)
