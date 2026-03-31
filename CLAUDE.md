# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Running the Pipeline

```bash
# Run on first 5 rows (default — safe for testing)
python pipeline.py

# Edit max_rows in pipeline.py to process more, or remove the limit entirely
```

Output goes to `output/structured.json` and `output/structured_flagged.json`.

## Generating the Report + Charts + Site

```bash
python report.py
```

This produces:
- `output/report.txt` — plain-text summary report
- `output/charts/top_rated_brands.png` — bar chart of avg rating per brand
- `output/charts/sentiment_by_brand.png` — stacked bar chart of sentiment per brand
- `output/charts/qa_flags_by_brand.png` — bar chart of QA flag counts per brand
- `index.html` — static site displaying the report and all three charts (hosted via GitHub Pages)

## Architecture

A single-file ETL pipeline (`pipeline.py`) that reads `amazon.csv`, sends each row to the Claude API via tool use, and writes validated structured JSON to `output/`.

**Data flow:**
1. `csv.DictReader` reads raw rows from `amazon.csv`
2. `extract_with_claude()` sends each row to `claude-haiku-4-5-20251001` using tool use (`tool_choice: forced`) to guarantee a structured JSON response matching the defined schema
3. `validate()` applies programmatic checks on top of Claude's own `qa_flags` field
4. Results are split: clean rows → `structured.json`, flagged rows → `structured_flagged.json`

**Why tool use instead of prompt-and-parse:**
Tool use forces Claude to return a schema-validated object. There is no regex parsing of free text — if the response doesn't match the schema, the API call fails loudly rather than silently producing bad data.

**Key schema fields extracted per product:**
- `brand` — inferred from product name
- `category_hierarchy` — split from pipe-delimited category string
- `pricing` — prices parsed to numbers (₹ stripped)
- `key_features` — split from pipe-delimited `about_product`
- `reviews[]` — reviewer, title, content paired by index + sentiment classification
- `qa_flags` — combined Claude-detected + programmatic data quality issues

## Project Context

This project was built as a learning exercise for a data science internship focused on **AI-driven data automation with verifiable pipelines**.

**What the project analyzes:**
- Raw Amazon electronics product listings (cables category) from `amazon.csv`
- Extracts brand, pricing, features, and review data from messy CSV rows
- Classifies each review sentiment (positive / neutral / negative)
- Flags data quality issues (mismatched review counts, placeholder names, copy-pasted reviews, etc.)
- Produces clean structured JSON, a summary report, and visualizations

**Dataset:** `amazon.csv` — columns: product_id, product_name, category, discounted_price, actual_price, discount_percentage, rating, rating_count, about_product, user_id, user_name, review_id, review_title, review_content, img_link, product_link.

**What makes this pipeline "verifiable":**
- Tool use forces schema compliance — bad output fails loudly, not silently
- `validate()` adds a second programmatic layer on top of Claude's output
- Flagged rows are written to a separate file for human review
- `max_rows` limit lets you spot-check output before running on the full dataset

## What This Project Demonstrates

End-to-end ETL pipeline — raw CSV data in, structured insight out:

- **Real API usage** — Claude Haiku via tool use (not just chat), forcing schema-validated output
- **Two-layer quality control** — Claude flags issues, then `validate()` adds a programmatic second pass; flagged and clean rows are split into separate files
- **Full reporting stack** — plain-text report, three matplotlib charts, and a static site all generated from the same JSON output
- **Public and presentable** — hosted on GitHub Pages, accessible to anyone without setup or API keys

## Public Site

The pipeline results are published as a static site via GitHub Pages (`index.html` at repo root).
No API key is needed to view the site — it only shows pre-generated output.

## Environment

- Python 3.13
- Dependencies: `anthropic`, `pandas`, `python-dotenv`, `matplotlib`
- API key goes in `.env` as `ANTHROPIC_API_KEY` (get one at console.anthropic.com)
- `.env` must never be committed — it is listed in `.gitignore`

## File Structure

```
pipeline.py          # ETL pipeline — reads CSV, calls Claude API, writes JSON
report.py            # Report + chart + site generator
amazon.csv           # Raw dataset
requirements.txt     # Python dependencies
.env.example         # Template for .env (safe to commit)
index.html           # Generated static site (GitHub Pages)
output/
  structured.json         # Clean + flagged products combined
  structured_flagged.json # Flagged products only
  report.txt              # Plain-text report
  charts/
    top_rated_brands.png
    sentiment_by_brand.png
    qa_flags_by_brand.png
```
