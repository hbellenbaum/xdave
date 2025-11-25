r"""
Scraping data from x-ray data booklet.
"""

import pandas as pd
import requests
import re
import csv

import tempfile
import pdfplumber
import os


def extract_electron_binding_energies():

    # Base URL and suffixes
    base_url = "https://xdb.lbl.gov/Section1/Table_1-1{}.htm"
    suffixes = ["a", "b", "c"]

    def clean_value(val):
        """Remove footnotes, stray unicode, and convert to float if possible."""
        if pd.isna(val):
            return None
        s = str(val)
        s = re.sub(r"[*†a-zA-Z]", "", s)  # remove footnote markers
        s = s.replace("\u0086", "").replace("\xa0", "")
        s = s.strip()
        try:
            return float(s)
        except ValueError:
            return s if s else None

    def is_duplicate_header_row(row):
        """
        Detect a repeated header row by checking if a row contains '1 2' in any of the first few cells.
        This matches the typical repeated header pattern in the tables.
        """
        row_text = ",".join(str(c) for c in row.values)
        # Check if it contains the string '1 2' or '2 21/2' which is in the repeated header
        return bool(re.search(r"1 2|2 21/2|3 23/2", row_text))

    # Collect cleaned tables
    all_tables = []

    for suf in suffixes:
        url = base_url.format(suf)

        response = requests.get(url)
        response.encoding = "latin1"

        df = pd.read_html(response.text, header=None)[0]

        # Drop duplicated header rows using the heuristic
        df = df[~df.apply(is_duplicate_header_row, axis=1)].reset_index(drop=True)

        # Clean each cell
        df = df.applymap(clean_value)

        all_tables.append(df)

    # Merge all three tables
    merged_df = pd.concat(all_tables, ignore_index=True)

    merged_df.columns = [
        "AN",
        "K 1s",
        "L1 2s",
        "L2 2p1/2",
        "L3 2p3/2",
        "M1 3s",
        "M2 3p1/2",
        "M3 3p3/2",
        "M4 3d3/2",
        "M5 3d5/2",
        "N1 4s",
        "N2 4p1/2",
        "N3 4p3/2",
    ]

    # Melt into long/tidy format
    long_df = merged_df.melt(id_vars=["AN"], var_name="Shell", value_name="Energy_eV")

    # Drop missing values (where there is no binding energy)
    long_df = long_df.dropna(subset=["Energy_eV"]).reset_index(drop=True)

    # Save
    long_df.to_csv("binding_energies_long.csv", index=False)
    long_df.to_json("binding_energies_long.json", orient="records", indent=2)

    # Save CSV/JSON
    merged_df.to_csv("binding_energies_full_clean_simple.csv", index=False)
    merged_df.to_json("binding_energies_full_clean_simple.json", orient="records", indent=2)


def extract_emission_line_data():
    PDF_URL = "https://xdb.lbl.gov/Section1/Table_1-2.pdf"
    output_csv = "emission_lines_table_1_2.csv"

    # Download PDF
    resp = requests.get(PDF_URL, timeout=30)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(resp.content)
    tmp.close()

    data = []

    try:
        with pdfplumber.open(tmp.name) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                for line in text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    # Match lines starting with a row number + element symbol
                    match = re.match(r"^\d+\s+([A-Z][a-z]?)\s+(.+)$", line)
                    if match:
                        element = match.group(1)
                        rest = match.group(2)
                        # Extract numeric values (allow commas)
                        numbers = re.findall(r"[\d,]+\.\d+|[\d,]+", rest)
                        numbers = [float(n.replace(",", "")) for n in numbers]
                        data.append([element] + numbers)

        # Determine max columns
        max_cols = max(len(row) for row in data)
        # Define standard emission line names for first 9 columns
        line_names = ["Kα1", "Kα2", "Kβ1", "Lα1", "Lα2", "Lβ1", "Lβ2", "Lγ1", "Mα1"]
        col_names = ["Element"] + line_names[: max_cols - 1]
        # Add generic names if there are extra columns
        if max_cols > len(line_names) + 1:
            extra_cols = max_cols - (len(line_names) + 1)
            col_names += [f"Col{i}" for i in range(1, extra_cols + 1)]

        df = pd.DataFrame(data, columns=col_names)

        # Save CSV
        df.to_csv(output_csv, index=False)
        return df

    finally:
        os.unlink(tmp.name)


def parse_line_data(line):
    # remove footnotes
    line = re.sub(r"[\*\†\‡]", "", line)
    # split on whitespace
    parts = line.strip().split()
    if len(parts) < 4:
        return None  # need at least Z, symbol, mass, density

    # first column: Z
    try:
        z = int(parts[0])
    except ValueError:
        return None

    # second column: name
    symbol = parts[1]

    def parse_number(s):
        s_clean = re.sub(r"[^0-9\.\-eE]", "", s)
        try:
            return float(s_clean)
        except ValueError:
            return None

    mass = parse_number(parts[2])
    dens = parse_number(parts[3])

    return {"atomic_number": z, "symbol": symbol, "atomic_mass": mass, "density": dens}


def scrape_pdf_with_symbol(pdf_path):
    data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split("\n")

            # search for header
            start = 0
            for i, line in enumerate(lines):
                if re.search(r"\bZ\b", line) and "Density" in line:
                    start = i + 1
                    break

            for line in lines[start:]:
                row = parse_line_data(line)
                if row:
                    data.append(row)
    return data


def extract_atomic_data():
    url = "https://xdb.lbl.gov/Section5/Table_5.2.pdf"
    print("Downloading PDF...")
    resp = requests.get(url)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    print("Extracting data from PDF...")
    data = scrape_pdf_with_symbol(tmp_path)

    output_file = os.path.dirname(__file__) + "/data/atomic_data_new.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["atomic_number", "symbol", "atomic_mass", "density"])
        writer.writeheader()
        writer.writerows(data)

    print(f"Data saved to: {output_file}")
    print(f"Extracted {len(data)} rows.")


if __name__ == "__main__":
    extract_atomic_data()
