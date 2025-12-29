# ICLR BibTeX Scraper

Generate a BibLaTeX-compatible `.bib` file from Curran Associates ICLR webtoc PDFs, with optional OpenReview URLs and ISBNs per year.

WARNING: vibe-coded.

## What it does

- Parses ICLR webtoc PDFs (old/new formats) and extracts titles, authors, and page ranges.
- Optionally joins OpenReview submissions to add forum URLs.
- Adds per-year ISBNs from a JSON mapping file.

## Requirements

- Python 3
- Dependencies:
  - `pymupdf`
  - `openreview-py` (only if using `--add-openreview-urls`)
  - `rapidfuzz` (optional, for faster fuzzy matching)

## Usage

```bash
python src/scrape_toc_to_bib.py <input_zip_or_dir> <output_bib>
```

Example:

```bash
python src/scrape_toc_to_bib.py data/iclr.zip iclr.bib --verbose --add-openreview-urls
```

## ISBN mapping

The script loads ISBNs from `data/iclr_isbn_by_year.json` by default. Override with:

```bash
python src/scrape_toc_to_bib.py ... --isbn-map /path/to/isbn_map.json
```

JSON format:

```json
{
  "2017": "978-1-7138-7271-9",
  "2018": "978-1-7138-7272-6"
}
```

## OpenReview notes

- Uses OpenReview API v1/v2 depending on year/invitation availability.
- If URL matching fails for a year, run with `--verbose` to see invitation probes.

## Output

BibTeX keys use the format:

```
<first_author_lastname>_<first_title_word>_<year>
```

