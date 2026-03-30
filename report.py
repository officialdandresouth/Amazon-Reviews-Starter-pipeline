import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
from pathlib import Path

matplotlib.use("Agg")  # non-interactive backend (no GUI window needed)

OUTPUT_PATH = "output/structured.json"
REPORT_PATH = "output/report.txt"
CHARTS_DIR = Path("output/charts")


def load_data(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten_reviews(data: list) -> pd.DataFrame:
    rows = []
    for product in data:
        for review in product.get("reviews", []):
            rows.append({
                "product_id": product["product_id"],
                "brand": product.get("brand", "Unknown"),
                "rating": product.get("rating"),
                "sentiment": review.get("sentiment"),
                "reviewer": review.get("reviewer"),
            })
    return pd.DataFrame(rows)


def build_products_df(data: list) -> pd.DataFrame:
    rows = []
    for p in data:
        pricing = p.get("pricing", {})
        rows.append({
            "product_id": p["product_id"],
            "brand": p.get("brand", "Unknown"),
            "rating": p.get("rating"),
            "rating_count": p.get("rating_count"),
            "discounted_price": pricing.get("discounted_price_inr"),
            "actual_price": pricing.get("actual_price_inr"),
            "discount_pct": pricing.get("discount_pct"),
            "num_features": len(p.get("key_features", [])),
            "num_reviews": len(p.get("reviews", [])),
            "num_flags": len(p.get("qa_flags", [])),
            "flagged": len(p.get("qa_flags", [])) > 0,
        })
    return pd.DataFrame(rows)


def report_section(title: str, lines: list) -> str:
    divider = "=" * 50
    return f"\n{divider}\n{title}\n{divider}\n" + "\n".join(lines)


def generate_report(data: list) -> str:
    products_df = build_products_df(data)
    reviews_df = flatten_reviews(data)
    sections = []

    # --- Overview ---
    total = len(products_df)
    flagged = products_df["flagged"].sum()
    sections.append(report_section("OVERVIEW", [
        f"Total products processed : {total}",
        f"Clean products           : {total - flagged}",
        f"Flagged products         : {flagged} ({flagged/total*100:.1f}%)",
        f"Total reviews extracted  : {len(reviews_df)}",
        f"Unique brands            : {products_df['brand'].nunique()}",
    ]))

    # --- Sentiment breakdown ---
    sentiment_counts = reviews_df["sentiment"].value_counts()
    total_reviews = len(reviews_df)
    sections.append(report_section("SENTIMENT BREAKDOWN", [
        f"{s:<10} {n:>6}  ({n/total_reviews*100:.1f}%)"
        for s, n in sentiment_counts.items()
    ]))

    # --- Sentiment by brand (top 10 brands by review count) ---
    top_brands = reviews_df["brand"].value_counts().head(10).index
    brand_sentiment = (
        reviews_df[reviews_df["brand"].isin(top_brands)]
        .groupby(["brand", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    lines = []
    for brand in brand_sentiment.index:
        row = brand_sentiment.loc[brand]
        total_b = row.sum()
        pos = row.get("positive", 0)
        neg = row.get("negative", 0)
        neu = row.get("neutral", 0)
        lines.append(f"{brand:<20} pos={pos:>3}  neu={neu:>3}  neg={neg:>3}  (total={total_b})")
    sections.append(report_section("SENTIMENT BY BRAND (top 10 by review volume)", lines))

    # --- Pricing ---
    valid_prices = products_df.dropna(subset=["discount_pct", "actual_price"])
    sections.append(report_section("PRICING SUMMARY", [
        f"Avg discount             : {valid_prices['discount_pct'].mean():.1f}%",
        f"Highest discount         : {valid_prices['discount_pct'].max():.0f}%",
        f"Lowest discount          : {valid_prices['discount_pct'].min():.0f}%",
        f"Avg actual price (INR)     : {valid_prices['actual_price'].mean():.0f}",
        f"Avg discounted price (INR) : {valid_prices['discounted_price'].mean():.0f}",
    ]))

    # --- Top rated brands (min 5 products) ---
    brand_ratings = (
        products_df.groupby("brand")
        .filter(lambda x: len(x) >= 5)
        .groupby("brand")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    lines = [f"{brand:<20} avg rating = {rating:.2f}" for brand, rating in brand_ratings.items()]
    sections.append(report_section("TOP RATED BRANDS (min 5 products)", lines))

    # --- Most flagged brands ---
    flag_by_brand = products_df.groupby("brand")["num_flags"].sum().sort_values(ascending=False).head(10)
    lines = [f"{brand:<20} {flags} flag(s)" for brand, flags in flag_by_brand.items()]
    sections.append(report_section("MOST QA FLAGS BY BRAND", lines))

    # --- Most common QA flag types ---
    all_flags = []
    for p in data:
        all_flags.extend(p.get("qa_flags", []))
    flag_counter = Counter(all_flags)
    lines = [f"{count:>4}x  {flag}" for flag, count in flag_counter.most_common(10)]
    sections.append(report_section("MOST COMMON QA FLAG TYPES (top 10)", lines))

    return "\n".join(sections) + "\n"


def generate_charts(data: list) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    products_df = build_products_df(data)
    reviews_df = flatten_reviews(data)

    # --- Chart 1: Top rated brands (min 5 products) ---
    brand_ratings = (
        products_df.groupby("brand")
        .filter(lambda x: len(x) >= 5)
        .groupby("brand")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(brand_ratings.index, brand_ratings.values, color="#4C72B0")
    ax.set_title("Top Rated Brands (min 5 products)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Rating")
    ax.set_ylim(0, 5)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "top_rated_brands.png", dpi=150)
    plt.close(fig)
    print("  Saved: output/charts/top_rated_brands.png")

    # --- Chart 2: Sentiment breakdown by brand (top 10 by review volume) ---
    top_brands = reviews_df["brand"].value_counts().head(10).index
    brand_sentiment = (
        reviews_df[reviews_df["brand"].isin(top_brands)]
        .groupby(["brand", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    sentiment_colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c", "mixed": "#f39c12"}
    colors = [sentiment_colors.get(col, "#aaaaaa") for col in brand_sentiment.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
    brand_sentiment.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Sentiment Breakdown by Brand (top 10 by review volume)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Reviews")
    ax.set_xlabel("")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "sentiment_by_brand.png", dpi=150)
    plt.close(fig)
    print("  Saved: output/charts/sentiment_by_brand.png")

    # --- Chart 3: Most QA-flagged brands ---
    flag_by_brand = (
        products_df.groupby("brand")["num_flags"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(flag_by_brand.index, flag_by_brand.values, color="#e74c3c")
    ax.set_title("Most QA-Flagged Brands (top 10)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total QA Flags")
    ax.bar_label(bars, padding=3, fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "qa_flags_by_brand.png", dpi=150)
    plt.close(fig)
    print("  Saved: output/charts/qa_flags_by_brand.png")


def generate_site(data: list) -> None:
    products_df = build_products_df(data)
    reviews_df = flatten_reviews(data)

    total = len(products_df)
    flagged = int(products_df["flagged"].sum())
    total_reviews = len(reviews_df)
    unique_brands = int(products_df["brand"].nunique())

    sentiment_counts = reviews_df["sentiment"].value_counts()
    pos = int(sentiment_counts.get("positive", 0))
    neu = int(sentiment_counts.get("neutral", 0))
    neg = int(sentiment_counts.get("negative", 0))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Amazon Electronics Pipeline Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f5; color: #222; margin: 0; padding: 0; }}
    header {{ background: #1a1a2e; color: white; padding: 2rem 2rem 1.5rem; }}
    header h1 {{ margin: 0 0 0.25rem; font-size: 1.6rem; }}
    header p {{ margin: 0; opacity: 0.7; font-size: 0.95rem; }}
    main {{ max-width: 960px; margin: 2rem auto; padding: 0 1.5rem; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .stat-card {{ background: white; border-radius: 8px; padding: 1.2rem 1rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center; }}
    .stat-card .value {{ font-size: 2rem; font-weight: 700; color: #1a1a2e; }}
    .stat-card .label {{ font-size: 0.8rem; color: #666; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .chart-section {{ margin-bottom: 2.5rem; }}
    .chart-section h2 {{ font-size: 1.1rem; margin-bottom: 0.75rem; color: #333; }}
    .chart-section img {{ width: 100%; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.1); background: white; }}
    .sentiment-pills {{ display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 2rem; }}
    .pill {{ padding: 0.5rem 1.2rem; border-radius: 999px; font-size: 0.9rem; font-weight: 600; }}
    .pill.pos {{ background: #d4edda; color: #155724; }}
    .pill.neu {{ background: #e2e3e5; color: #383d41; }}
    .pill.neg {{ background: #f8d7da; color: #721c24; }}
    footer {{ text-align: center; padding: 2rem; color: #999; font-size: 0.8rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Amazon Electronics — Pipeline Report</h1>
    <p>AI-driven data extraction and quality analysis · Claude API (Haiku) · {total} products</p>
  </header>
  <main>
    <div class="stats">
      <div class="stat-card"><div class="value">{total}</div><div class="label">Products</div></div>
      <div class="stat-card"><div class="value">{total - flagged}</div><div class="label">Clean</div></div>
      <div class="stat-card"><div class="value">{flagged}</div><div class="label">QA Flagged</div></div>
      <div class="stat-card"><div class="value">{total_reviews}</div><div class="label">Reviews</div></div>
      <div class="stat-card"><div class="value">{unique_brands}</div><div class="label">Brands</div></div>
    </div>

    <h2 style="margin-bottom:0.5rem;">Overall Sentiment</h2>
    <div class="sentiment-pills">
      <span class="pill pos">Positive {pos} ({pos/total_reviews*100:.1f}%)</span>
      <span class="pill neu">Neutral {neu} ({neu/total_reviews*100:.1f}%)</span>
      <span class="pill neg">Negative {neg} ({neg/total_reviews*100:.1f}%)</span>
    </div>

    <div class="chart-section">
      <h2>Top Rated Brands</h2>
      <img src="output/charts/top_rated_brands.png" alt="Top rated brands chart" />
    </div>
    <div class="chart-section">
      <h2>Sentiment Breakdown by Brand</h2>
      <img src="output/charts/sentiment_by_brand.png" alt="Sentiment by brand chart" />
    </div>
    <div class="chart-section">
      <h2>Most QA-Flagged Brands</h2>
      <img src="output/charts/qa_flags_by_brand.png" alt="QA flags by brand chart" />
    </div>
  </main>
  <footer>Built with Claude API · Source on GitHub</footer>
</body>
</html>"""

    site_path = Path("index.html")
    site_path.write_text(html, encoding="utf-8")
    print(f"  Saved: index.html")


def main():
    if not Path(OUTPUT_PATH).exists():
        print(f"Output file not found: {OUTPUT_PATH}")
        print("Run pipeline.py first.")
        return

    print(f"Loading {OUTPUT_PATH}...")
    data = load_data(OUTPUT_PATH)
    print(f"Loaded {len(data)} products. Generating report...")

    report = generate_report(data)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(report.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nReport saved -> {REPORT_PATH}")

    print("\nGenerating charts...")
    generate_charts(data)
    print("Charts saved -> output/charts/")

    print("\nGenerating site...")
    generate_site(data)
    print("Site saved -> index.html")


if __name__ == "__main__":
    main()
