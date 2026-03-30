import anthropic
import csv
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Tool definition forces Claude to return a strict JSON schema
EXTRACT_TOOL = {
    "name": "extract_product_data",
    "description": "Extract structured product information from an Amazon product row",
    "input_schema": {
        "type": "object",
        "properties": {
            "product_id": {"type": "string"},
            "brand": {
                "type": "string",
                "description": "Brand name extracted from product name"
            },
            "product_name": {"type": "string"},
            "category_hierarchy": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Category levels split by | from broad to specific"
            },
            "pricing": {
                "type": "object",
                "properties": {
                    "discounted_price_inr": {"type": "number"},
                    "actual_price_inr": {"type": "number"},
                    "discount_pct": {"type": "number"}
                },
                "required": ["discounted_price_inr", "actual_price_inr", "discount_pct"]
            },
            "rating": {"type": "number"},
            "rating_count": {"type": "integer"},
            "key_features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key features from about_product, each cleaned and concise"
            },
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "reviewer": {"type": "string"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "neutral", "negative"]
                        }
                    },
                    "required": ["reviewer", "title", "content", "sentiment"]
                }
            },
            "qa_flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Data quality issues found (e.g. image URLs in review text, missing fields, mismatched review counts)"
            }
        },
        "required": [
            "product_id", "brand", "product_name", "category_hierarchy",
            "pricing", "rating", "rating_count", "key_features", "reviews", "qa_flags"
        ]
    }
}


def extract_with_claude(row: dict) -> dict:
    """Send one CSV row to Claude and return structured data via tool use."""
    prompt = f"""Extract structured information from this Amazon product record.

Product ID: {row['product_id']}
Name: {row['product_name']}
Category: {row['category']}
Discounted Price: {row['discounted_price']}
Actual Price: {row['actual_price']}
Discount: {row['discount_percentage']}
Rating: {row['rating']}
Rating Count: {row['rating_count']}
About Product: {row['about_product']}
Reviewers: {row['user_name']}
Review Titles: {row['review_title']}
Review Content: {row['review_content']}

Instructions:
- Strip ₹ and commas from prices, return as numbers
- Split category by | into hierarchy array
- Split about_product by | into key_features list
- Pair reviewers, titles, and content by comma-separated index
- Classify each review sentiment as positive, neutral, or negative
- Add qa_flags for any quality issues (image URLs in review text, blank fields, count mismatches)"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_product_data"},
        messages=[{"role": "user", "content": prompt}]
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise ValueError(f"No tool_use block in response for product {row['product_id']}")


def validate(data: dict) -> list:
    """Additional programmatic checks on top of Claude's qa_flags."""
    issues = []
    pricing = data.get("pricing", {})

    if not data.get("brand"):
        issues.append("brand_missing")

    discounted = pricing.get("discounted_price_inr", 0)
    actual = pricing.get("actual_price_inr", 0)
    if actual > 0 and discounted > actual:
        issues.append("discounted_price_exceeds_actual")

    if not data.get("key_features"):
        issues.append("no_features_extracted")

    if not data.get("reviews"):
        issues.append("no_reviews_extracted")

    return issues


def run_pipeline(input_path: str, output_path: str, max_rows: int = None):
    Path("output").mkdir(exist_ok=True)

    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if max_rows:
        rows = rows[:max_rows]

    print(f"Processing {len(rows)} products...\n")

    results = []
    flagged = []

    for i, row in enumerate(rows):
        product_id = row.get("product_id", f"row_{i}")
        print(f"[{i+1}/{len(rows)}] {product_id}", end=" ... ", flush=True)

        try:
            structured = extract_with_claude(row)
            validation_issues = validate(structured)

            # Merge Claude's qa_flags with programmatic validation issues
            all_flags = list(set(structured.get("qa_flags", []) + validation_issues))
            structured["qa_flags"] = all_flags

            if all_flags:
                flagged.append(structured)
                print(f"FLAGGED ({', '.join(all_flags)})")
            else:
                print("OK")

            results.append(structured)

        except Exception as e:
            print(f"ERROR — {e}")
            flagged.append({"product_id": product_id, "_error": str(e)})

    # Write all structured results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{len(results)} products written -> {output_path}")

    # Write flagged rows separately for review
    if flagged:
        flagged_path = output_path.replace(".json", "_flagged.json")
        with open(flagged_path, "w", encoding="utf-8") as f:
            json.dump(flagged, f, indent=2, ensure_ascii=False)
        print(f"{len(flagged)} flagged rows -> {flagged_path}")


if __name__ == "__main__":
    run_pipeline(
        input_path="amazon.csv",
        output_path="output/structured.json",
        max_rows=100
    )
