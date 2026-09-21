# Data Boundary

The portfolio repository does not ship raw social-media exports, processed comments, author identifiers or row-level predictions.

To run the complete pipeline locally, place permitted CSV/XLSX exports in `data/raw/scraped/`. The source files are ignored by Git. If no files are present, the pipeline uses `src/make_sample_data.py` to generate a privacy-safe demonstration dataset.

The tracked files under `data/dict/` are generic vocabulary and stopword resources only.
