import pandas as pd
from jinja2 import Environment, FileSystemLoader
import argparse
import os
import sys

def generate_xml(csv_path, template_path, output_path, currency):
    # 1. Load Data
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, dtype=str)
        # Convert NaN to None so Jinja handles them gracefully
        records = df.where(pd.notnull(df), None).to_dict(orient='records')
    except Exception as e:
        sys.exit(f"Error reading CSV: {e}")

    # 2. Setup Template Environment
    print(f"Loading template {template_path}...")
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path) or '.'))
    template = env.get_template(os.path.basename(template_path))

    # 3. Render
    print(f"Rendering {len(records)} records...")
    xml_content = template.render(records=records, currency=currency)

    # 4. Write Output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    print(f"Success! XML written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--template", default="esma_template.xml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--currency", default="GBP")
    args = parser.parse_args()

    generate_xml(args.input, args.template, args.output, args.currency)