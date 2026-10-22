"""
=============================================================================
01_explore_dataset.py
=============================================================================
Phase 1 – Dataset Analysis and Preprocessing Planning
Project: AI-Based Startup Viability Prediction and Validation System

PURPOSE
-------
This script performs Exploratory Data Analysis (EDA) on the raw startup
dataset WITHOUT modifying the original CSV file. All cleaned or intermediate
outputs are saved to the /outputs and /data directories.

HOW TO RUN
----------
    python 01_explore_dataset.py

    Optionally override the CSV path:
    python 01_explore_dataset.py --csv "path/to/your_file.csv"

OUTPUT
------
    outputs/plots/         → all charts (PNG)
    outputs/reports/       → eda_summary.txt (text-based EDA report)
    data/                  → intermediate datasets (labeled, no operating rows)

IMPORTANT
---------
- The original CSV is NEVER modified.
- No ML model is trained here.
- No fake data is generated.
- Status = "operating" companies are kept SEPARATE; they are NOT labeled.
=============================================================================
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")   # suppress minor pandas/matplotlib warnings

# ---------------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------------

# Default path – relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "big_startup_secsees_dataset.csv")

# Output directories
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "outputs", "reports")
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")

for directory in [PLOTS_DIR, REPORTS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# We will also write a running text report to a file
REPORT_PATH = os.path.join(REPORTS_DIR, "eda_summary.txt")

# Reasonable year bounds for date sanity checks
MIN_FOUNDING_YEAR = 1800   # anything older is suspicious
MAX_ALLOWED_YEAR  = 2023   # data was collected ~2023; future dates are invalid

# ---------------------------------------------------------------------------
# HELPER UTILITIES
# ---------------------------------------------------------------------------

# We write every print() call to both the console AND to the report file.
report_lines = []

def rprint(*args, **kwargs):
    """Print to console AND collect into report_lines for file export."""
    text = " ".join(str(a) for a in args)
    print(text, **kwargs)
    report_lines.append(text)

def section(title):
    """Print a prominent section header."""
    bar = "=" * 70
    rprint(f"\n{bar}")
    rprint(f"  {title}")
    rprint(bar)

def subsection(title):
    """Print a minor subsection divider."""
    rprint(f"\n--- {title} ---")

def save_figure(fig, filename):
    """Save a matplotlib figure to the plots directory."""
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    rprint(f"  [PLOT SAVED] {path}")

# ---------------------------------------------------------------------------
# ARGUMENT PARSING – allows overriding the CSV path from command line
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="EDA script for AI Startup Viability Prediction project."
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"Path to the dataset CSV file. Default: {DEFAULT_CSV}"
    )
    return parser.parse_args()

# ===========================================================================
# SECTION 1 – LOAD THE CSV
# ===========================================================================

def load_data(csv_path):
    section("SECTION 1 – LOAD THE DATASET")

    rprint(f"  CSV path : {csv_path}")

    if not os.path.exists(csv_path):
        rprint(f"  ERROR: File not found at '{csv_path}'")
        sys.exit(1)

    # encoding='utf-8' handles international characters (e.g. Chinese, European).
    # low_memory=False forces pandas to read all rows before deciding column types,
    # which prevents wrong dtype inference on mixed-type columns.
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    rprint(f"  File loaded successfully.")
    return df

# ===========================================================================
# SECTION 2 – BASIC SHAPE AND SCHEMA
# ===========================================================================

def basic_info(df):
    section("SECTION 2 – BASIC SHAPE AND SCHEMA")

    subsection("Dataset Dimensions")
    rprint(f"  Rows    : {df.shape[0]:,}")
    rprint(f"  Columns : {df.shape[1]}")

    subsection("Column Names")
    for i, col in enumerate(df.columns, 1):
        rprint(f"  {i:>2}. {col}")

    subsection("Data Types (as inferred by pandas)")
    rprint(f"  {'Column':<25} {'Dtype':<15}")
    rprint(f"  {'-'*25} {'-'*15}")
    for col, dtype in df.dtypes.items():
        rprint(f"  {col:<25} {str(dtype):<15}")

    rprint("\n  NOTE: pandas reads all columns as 'object' (string) unless")
    rprint("  values are clearly numeric. Date columns will appear as 'object'")
    rprint("  here – we parse them explicitly in Section 7.")

    return df

# ===========================================================================
# SECTION 3 – STATUS COLUMN ANALYSIS AND BINARY TARGET CREATION
# ===========================================================================

def analyze_status(df):
    section("SECTION 3 – STATUS COLUMN ANALYSIS AND BINARY TARGET CREATION")

    # 3a. Count each status value
    subsection("Raw Status Counts")
    status_counts = df["status"].value_counts(dropna=False)
    total = len(df)
    rprint(f"  {'Status':<15} {'Count':>8}  {'Percentage':>10}")
    rprint(f"  {'-'*15} {'-'*8}  {'-'*10}")
    for status, count in status_counts.items():
        pct = count / total * 100
        rprint(f"  {str(status):<15} {count:>8,}  {pct:>9.2f}%")

    # 3b. Explain what each status means in business terms
    subsection("Status Definitions for This Project")
    rprint("  acquired  → company was bought by another company           → SUCCESS (label = 1)")
    rprint("  ipo       → company went public on stock exchange           → SUCCESS (label = 1)")
    rprint("  closed    → company shut down / went out of business        → FAILURE (label = 0)")
    rprint("  operating → company is still running at time of data collection")
    rprint("              → UNKNOWN outcome; EXCLUDED from training dataset")
    rprint("  (NaN)     → status not recorded                             → EXCLUDED")

    # 3c. Create binary target – only for companies with a known definitive outcome
    #
    # WHY we exclude 'operating':
    #   A company labeled "operating" in the dataset might later be acquired,
    #   go IPO, or close. We do NOT know its final outcome. Using it as
    #   success OR failure would introduce incorrect labels into training.
    #   This is a fundamental data integrity rule, not just caution.
    #
    known_mask = df["status"].isin(["acquired", "ipo", "closed"])
    df_known   = df[known_mask].copy()
    df_operating = df[df["status"] == "operating"].copy()
    df_other     = df[~known_mask & (df["status"] != "operating")].copy()

    # Assign binary label: 1 = success, 0 = failure
    df_known["target"] = df_known["status"].apply(
        lambda s: 1 if s in ["acquired", "ipo"] else 0
    )

    subsection("Binary Target Distribution (Training-Eligible Companies Only)")
    target_counts = df_known["target"].value_counts()
    rprint(f"  Label 0 (FAILURE – closed)           : {target_counts.get(0, 0):>7,}")
    rprint(f"  Label 1 (SUCCESS – acquired or ipo)  : {target_counts.get(1, 0):>7,}")
    rprint(f"  Total training-eligible rows         : {len(df_known):>7,}")
    rprint(f"  Operating (excluded from training)   : {len(df_operating):>7,}")
    rprint(f"  Other / NaN status (excluded)        : {len(df_other):>7,}")

    # Class imbalance check
    if 0 in target_counts.index and 1 in target_counts.index:
        ratio = target_counts[0] / target_counts[1]
        rprint(f"\n  Class ratio (failure : success)      : {ratio:.2f} : 1")
        if ratio > 2:
            rprint("  ⚠ WARNING: Significant class imbalance detected.")
            rprint("    During model training, techniques like SMOTE, class_weight,")
            rprint("    or stratified sampling should be considered.")

    # 3d. Plot status distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Status Distribution", fontsize=14, fontweight="bold")

    # Full dataset status breakdown
    sc = status_counts.reset_index()
    sc.columns = ["status", "count"]
    sns.barplot(data=sc, x="status", y="count", ax=axes[0], hue="status", palette="Set2", legend=False)
    axes[0].set_title("All Companies – Status Counts")
    axes[0].set_xlabel("Status")
    axes[0].set_ylabel("Count")
    for bar in axes[0].patches:
        axes[0].annotate(f'{int(bar.get_height()):,}',
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha='center', va='bottom', fontsize=9)

    # Binary target (known only)
    target_labels = ["Failure (0)", "Success (1)"]
    target_vals   = [target_counts.get(0, 0), target_counts.get(1, 0)]
    axes[1].pie(target_vals, labels=target_labels, autopct="%1.1f%%",
                colors=["#e74c3c", "#2ecc71"], startangle=140)
    axes[1].set_title("Binary Target – Known Outcomes Only")

    save_figure(fig, "03_status_distribution.png")

    return df_known, df_operating

# ===========================================================================
# SECTION 4 – MISSING VALUE ANALYSIS
# ===========================================================================

def missing_values(df, df_known):
    section("SECTION 4 – MISSING VALUE ANALYSIS")

    def show_missing(data, label):
        subsection(f"Missing Values – {label}")
        total_rows = len(data)
        rprint(f"  {'Column':<25} {'Missing':>8}  {'% Missing':>10}  {'Status'}")
        rprint(f"  {'-'*25} {'-'*8}  {'-'*10}  {'-'*20}")
        for col in data.columns:
            n_missing = data[col].isna().sum()
            pct = n_missing / total_rows * 100
            severity = "OK" if pct == 0 else ("LOW" if pct < 5 else ("MODERATE" if pct < 30 else "HIGH"))
            rprint(f"  {col:<25} {n_missing:>8,}  {pct:>9.2f}%  {severity}")

    show_missing(df,       "Full Dataset (all rows)")
    show_missing(df_known, "Training Dataset (known outcomes only)")

    # Heatmap of missing values (full dataset, first 5000 rows for visibility)
    fig, ax = plt.subplots(figsize=(14, 5))
    sample = df.iloc[:5000].isnull()
    sns.heatmap(sample.T, cmap="YlOrRd", cbar=True, ax=ax,
                yticklabels=True, xticklabels=False)
    ax.set_title("Missing Value Heatmap – First 5,000 Rows (Yellow = Missing)", fontsize=12)
    ax.set_xlabel("Rows →")
    ax.set_ylabel("Columns")
    save_figure(fig, "04_missing_values_heatmap.png")

# ===========================================================================
# SECTION 5 – DUPLICATE RECORDS
# ===========================================================================

def duplicates(df):
    section("SECTION 5 – DUPLICATE RECORDS")

    # 5a. Exact duplicate rows – every column matches completely
    exact_dupes = df.duplicated().sum()
    rprint(f"\n  Exact duplicate rows (all 14 columns identical) : {exact_dupes:,}")
    if exact_dupes > 0:
        rprint("  → These are safe to drop during preprocessing.")
        rprint("  → We report them here but do NOT delete them from the CSV.")

    # 5b. Duplicate company names
    #   A company name appearing twice could mean:
    #   (a) a genuine duplicate entry (data error) – safe to deduplicate
    #   (b) two different companies with the same name (e.g. "Atlas")
    #       – NOT safe to deduplicate without checking the permalink
    name_counts = df["name"].value_counts()
    dup_names   = name_counts[name_counts > 1]
    rprint(f"\n  Companies with duplicate names                   : {len(dup_names):,}")
    rprint(f"  Total rows involved in name duplicates           : {dup_names.sum():,}")

    subsection("Top 15 Most-Duplicated Company Names")
    rprint(f"  {'Company Name':<45} {'Occurrences':>12}")
    rprint(f"  {'-'*45} {'-'*12}")
    for name_val, cnt in dup_names.head(15).items():
        rprint(f"  {str(name_val):<45} {cnt:>12,}")

    # 5c. Cross-check: do duplicated names have different permalinks?
    #   permalink is a URL slug that should be globally unique in Crunchbase.
    #   If two rows share a name but differ in permalink → they are likely
    #   two distinct companies that happen to share a name.
    subsection("Permalink Uniqueness Check")
    dup_permalinks = df["permalink"].value_counts()
    dup_pl = dup_permalinks[dup_permalinks > 1]
    rprint(f"  Duplicate permalink values (true data duplicates): {len(dup_pl):,}")
    if len(dup_pl) > 0:
        rprint("  → Rows with the same permalink are genuine duplicates – safe to drop the extras.")

    rprint("\n  DECISION: Do NOT automatically delete duplicate names.")
    rprint("  Reason: Same name ≠ same company. Use permalink as the true unique ID.")

# ===========================================================================
# SECTION 6 – FUNDING_TOTAL_USD INSPECTION
# ===========================================================================

def inspect_funding(df):
    section("SECTION 6 – FUNDING_TOTAL_USD INSPECTION")

    raw = df["funding_total_usd"].copy()

    # 6a. Count non-numeric values BEFORE conversion
    #   pd.to_numeric with errors='coerce' converts anything it cannot parse
    #   into NaN. We can then see how many were originally non-numeric.
    subsection("Non-Numeric Values in funding_total_usd")
    converted = pd.to_numeric(raw, errors="coerce")
    original_nulls    = raw.isna().sum()
    after_nulls       = converted.isna().sum()
    newly_nullified   = after_nulls - original_nulls

    rprint(f"  Originally missing (blank)               : {original_nulls:,}")
    rprint(f"  Became NaN after numeric conversion      : {after_nulls:,}")
    rprint(f"  Newly nullified (were non-numeric text)  : {newly_nullified:,}")

    # Show sample of non-numeric values
    non_numeric_mask = raw.notna() & converted.isna()
    non_numeric_vals = raw[non_numeric_mask].value_counts().head(20)
    if len(non_numeric_vals) > 0:
        rprint(f"\n  Sample of non-numeric values found:")
        rprint(f"  {'Value':<20} {'Count':>8}")
        rprint(f"  {'-'*20} {'-'*8}")
        for val, cnt in non_numeric_vals.items():
            rprint(f"  {str(val):<20} {cnt:>8,}")

    # 6b. Descriptive statistics on the cleaned numeric column
    subsection("Descriptive Statistics (after numeric conversion)")
    stats = converted.describe()
    rprint(f"  Count (non-null)  : {stats['count']:>15,.0f}")
    rprint(f"  Mean              : ${stats['mean']:>15,.2f}")
    rprint(f"  Std Dev           : ${stats['std']:>15,.2f}")
    rprint(f"  Min               : ${stats['min']:>15,.2f}")
    rprint(f"  25th percentile   : ${stats['25%']:>15,.2f}")
    rprint(f"  Median (50th)     : ${stats['50%']:>15,.2f}")
    rprint(f"  75th percentile   : ${stats['75%']:>15,.2f}")
    rprint(f"  Max               : ${stats['max']:>15,.2f}")

    # 6c. Outlier inspection
    #   The Interquartile Range (IQR) method: values below Q1 - 1.5*IQR
    #   or above Q3 + 1.5*IQR are considered potential outliers.
    subsection("Outlier Inspection (IQR Method)")
    q1  = converted.quantile(0.25)
    q3  = converted.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers    = converted[(converted < lower_fence) | (converted > upper_fence)]
    rprint(f"  Q1 (25th pct)     : ${q1:>15,.2f}")
    rprint(f"  Q3 (75th pct)     : ${q3:>15,.2f}")
    rprint(f"  IQR               : ${iqr:>15,.2f}")
    rprint(f"  Lower fence       : ${lower_fence:>15,.2f}")
    rprint(f"  Upper fence       : ${upper_fence:>15,.2f}")
    rprint(f"  Outlier count     : {len(outliers):>15,}")
    rprint(f"  % of total rows   : {len(outliers)/len(df)*100:>14.2f}%")

    # 6d. Top 10 highest funding values
    subsection("Top 10 Highest Funding Values")
    top10 = converted.nlargest(10)
    for rank, (idx, val) in enumerate(top10.items(), 1):
        name = df.loc[idx, "name"] if "name" in df.columns else idx
        rprint(f"  {rank:>2}. {str(name):<40} ${val:>20,.0f}")

    # 6e. Plot: distribution of funding (log scale for readability)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("funding_total_usd Distribution", fontsize=14, fontweight="bold")

    # Histogram on log scale (log because funding spans many orders of magnitude)
    valid_funding = converted.dropna()
    valid_positive = valid_funding[valid_funding > 0]
    axes[0].hist(np.log10(valid_positive + 1), bins=60, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].set_title("Funding Distribution (log10 scale)")
    axes[0].set_xlabel("log10(funding_total_usd)")
    axes[0].set_ylabel("Number of Companies")

    # Box plot to show spread and outliers
    axes[1].boxplot(valid_positive, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#3498db", alpha=0.6))
    axes[1].set_title("Box Plot (raw values – outliers visible)")
    axes[1].set_ylabel("funding_total_usd (USD)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    save_figure(fig, "06_funding_distribution.png")

    return converted

# ===========================================================================
# SECTION 7 – DATE COLUMN INSPECTION
# ===========================================================================

def inspect_dates(df):
    section("SECTION 7 – DATE COLUMN INSPECTION")

    date_cols = ["founded_at", "first_funding_at", "last_funding_at"]

    for col in date_cols:
        subsection(f"Column: {col}")
        raw = df[col].copy()

        # Convert to datetime – errors='coerce' turns unparseable strings into NaT
        parsed = pd.to_datetime(raw, errors="coerce")

        original_nulls  = raw.isna().sum()
        after_nulls     = parsed.isna().sum()
        newly_nat       = after_nulls - original_nulls

        rprint(f"  Originally missing             : {original_nulls:,}")
        rprint(f"  NaT after parsing              : {after_nulls:,}")
        rprint(f"  Newly unparseable (bad strings): {newly_nat:,}")

        valid = parsed.dropna()
        if len(valid) > 0:
            rprint(f"  Earliest date                  : {valid.min().date()}")
            rprint(f"  Latest date                    : {valid.max().date()}")

            # Check for dates in the far future (> MAX_ALLOWED_YEAR)
            future = valid[valid.dt.year > MAX_ALLOWED_YEAR]
            rprint(f"  Dates after {MAX_ALLOWED_YEAR} (suspicious) : {len(future):,}")
            if len(future) > 0:
                rprint(f"  Sample suspicious future dates:")
                for idx in future.head(5).index:
                    rprint(f"    Row {idx}: {df.loc[idx, col]}  (company: {df.loc[idx, 'name']})")

            # Check for very old dates (< MIN_FOUNDING_YEAR)
            very_old = valid[valid.dt.year < MIN_FOUNDING_YEAR]
            rprint(f"  Dates before {MIN_FOUNDING_YEAR} (suspicious): {len(very_old):,}")

            # Year distribution
            year_dist = valid.dt.year.value_counts().sort_index()
            rprint(f"\n  Year range with most activity:")
            # Show only top 15 years by count
            top_years = year_dist.nlargest(15).sort_index()
            for yr, cnt in top_years.items():
                bar = "█" * min(int(cnt / max(year_dist) * 40), 40)
                rprint(f"    {yr}: {cnt:>6,}  {bar}")

    # Plot: Year distributions for all three date columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Date Column – Year Distributions", fontsize=14, fontweight="bold")

    for i, col in enumerate(date_cols):
        parsed = pd.to_datetime(df[col], errors="coerce")
        years  = parsed.dt.year.dropna()
        # Focus on plausible range (1990–2023) for readability
        years_filtered = years[(years >= 1990) & (years <= MAX_ALLOWED_YEAR)]
        axes[i].hist(years_filtered, bins=34, color="#9b59b6", edgecolor="white", alpha=0.8)
        axes[i].set_title(f"{col}\n(1990–{MAX_ALLOWED_YEAR})")
        axes[i].set_xlabel("Year")
        axes[i].set_ylabel("Count")

    save_figure(fig, "07_date_distributions.png")

# ===========================================================================
# SECTION 8 – CATEGORY_LIST ANALYSIS
# ===========================================================================

def analyze_categories(df):
    section("SECTION 8 – CATEGORY_LIST ANALYSIS")

    raw = df["category_list"].dropna()

    rprint(f"  Rows with at least one category : {len(raw):,}")
    rprint(f"  Rows with no category (missing) : {df['category_list'].isna().sum():,}")

    # category_list uses pipe "|" as a separator, e.g. "Mobile|Apps|Games"
    # We split each row on "|" and collect all individual categories
    all_cats = []
    for entry in raw:
        parts = [c.strip() for c in str(entry).split("|") if c.strip()]
        all_cats.extend(parts)

    cat_series  = pd.Series(all_cats)
    unique_cats = cat_series.nunique()
    rprint(f"\n  Total category mentions (with repeats) : {len(all_cats):,}")
    rprint(f"  Unique individual categories            : {unique_cats:,}")

    # 8b. Top 30 most common categories
    subsection("Top 30 Most Common Individual Categories")
    top_cats = cat_series.value_counts().head(30)
    rprint(f"  {'Category':<35} {'Count':>8}  {'% of Categorized Rows':>22}")
    rprint(f"  {'-'*35} {'-'*8}  {'-'*22}")
    for cat, cnt in top_cats.items():
        pct = cnt / len(raw) * 100
        rprint(f"  {cat:<35} {cnt:>8,}  {pct:>21.1f}%")

    # 8c. How many categories does each company have?
    cats_per_company = df["category_list"].dropna().apply(
        lambda x: len([c for c in str(x).split("|") if c.strip()])
    )
    rprint(f"\n  Categories per company:")
    rprint(f"    Min    : {cats_per_company.min()}")
    rprint(f"    Max    : {cats_per_company.max()}")
    rprint(f"    Median : {cats_per_company.median()}")
    rprint(f"    Mean   : {cats_per_company.mean():.2f}")

    # 8d. Explain ML transformation strategies
    subsection("How to Transform category_list into ML Features (Future Phase)")
    rprint("""
  The model cannot consume raw text like "Mobile|Apps|Games" directly.
  Three approaches are commonly used:

  Option A – Multi-hot encoding (Binary flags):
    Create one binary column per category (1 = company is in that category).
    E.g.:  col_Mobile=1, col_Apps=1, col_Games=1, col_SaaS=0, ...
    PRO: Simple, interpretable.
    CON: With ~1000+ categories, this creates very wide (sparse) feature sets.
    RECOMMENDED: Apply only to the top N (e.g. top 50) categories.

  Option B – Category grouping (Macro-sectors):
    Map individual categories to broader sectors:
    "Mobile", "Apps", "Games" → "Consumer Technology"
    "SaaS", "Enterprise Software" → "B2B Software"
    PRO: Fewer columns, reduces sparsity.
    CON: Requires manual mapping effort.

  Option C – Frequency encoding:
    Replace category with how common it is in successful companies.
    E.g.:  category_success_rate = success_count / total_count for that category.
    PRO: Single numeric column, captures signal.
    CON: Requires careful cross-validation to avoid leakage.

  RECOMMENDED FOR THIS PROJECT:
    Use Option A with top 30–50 categories, combined with a
    "number of categories" numeric feature.
  """)

    # Plot: Top 20 categories bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    top20 = top_cats.head(20)
    sns.barplot(x=top20.values, y=top20.index, ax=ax,
                hue=top20.index, palette="viridis", legend=False)
    ax.set_title("Top 20 Most Frequent Startup Categories", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Companies")
    ax.set_ylabel("Category")
    save_figure(fig, "08_top_categories.png")

# ===========================================================================
# SECTION 9 – GEOGRAPHIC COLUMNS
# ===========================================================================

def analyze_geography(df):
    section("SECTION 9 – GEOGRAPHIC COLUMN ANALYSIS")

    geo_cols = ["country_code", "state_code", "region", "city"]

    for col in geo_cols:
        subsection(f"Column: {col}")
        n_missing = df[col].isna().sum()
        n_unique  = df[col].nunique()
        pct_missing = n_missing / len(df) * 100
        rprint(f"  Missing values : {n_missing:,}  ({pct_missing:.1f}%)")
        rprint(f"  Unique values  : {n_unique:,}")

        top10 = df[col].value_counts().head(10)
        rprint(f"  Top 10 values:")
        rprint(f"  {'Value':<25} {'Count':>8}  {'%':>6}")
        rprint(f"  {'-'*25} {'-'*8}  {'-'*6}")
        for val, cnt in top10.items():
            pct = cnt / len(df) * 100
            rprint(f"  {str(val):<25} {cnt:>8,}  {pct:>5.1f}%")

    # Plot: Top 20 countries by startup count
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Geographic Distribution", fontsize=14, fontweight="bold")

    top_countries = df["country_code"].value_counts().head(20)
    sns.barplot(x=top_countries.values, y=top_countries.index,
                ax=axes[0], hue=top_countries.index, palette="coolwarm", legend=False)
    axes[0].set_title("Top 20 Countries by Startup Count")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Country Code")

    # USA breakdown by region
    usa_df = df[df["country_code"] == "USA"]
    top_regions = usa_df["region"].value_counts().head(15)
    if len(top_regions) > 0:
        sns.barplot(x=top_regions.values, y=top_regions.index,
                    ax=axes[1], hue=top_regions.index, palette="Set1", legend=False)
        axes[1].set_title("Top 15 USA Regions")
        axes[1].set_xlabel("Count")
        axes[1].set_ylabel("Region")
    else:
        axes[1].set_visible(False)

    save_figure(fig, "09_geographic_distribution.png")

    # Geographic ML recommendation
    subsection("Geographic ML Strategy")
    rprint("""
  country_code : High cardinality (~140+ unique) but useful.
    → Target-encode or use top-N countries + "Other" bucket.
    → 'USA' dominates the dataset; a binary is_usa flag may suffice.

  state_code   : Only meaningful for USA companies; sparse for others.
    → Conditionally useful for USA-only sub-model; otherwise drop.

  region       : Meaningful only within a country context.
    → If using, restrict to USA companies; too sparse globally.

  city         : Very high cardinality (~10,000+ unique cities).
    → Likely too sparse for direct use. Drop or aggregate to region.
  """)

# ===========================================================================
# SECTION 10 – FUNDING_ROUNDS INSPECTION
# ===========================================================================

def inspect_funding_rounds(df):
    section("SECTION 10 – FUNDING_ROUNDS INSPECTION")

    col = "funding_rounds"
    rprint(f"  Missing values : {df[col].isna().sum():,}")

    converted = pd.to_numeric(df[col], errors="coerce")
    rprint(f"  Non-numeric    : {converted.isna().sum() - df[col].isna().sum():,}")

    stats = converted.describe()
    rprint(f"  Min            : {stats['min']:.0f}")
    rprint(f"  Max            : {stats['max']:.0f}")
    rprint(f"  Median         : {stats['50%']:.0f}")
    rprint(f"  Mean           : {stats['mean']:.2f}")

    val_counts = converted.value_counts().sort_index()
    rprint(f"\n  Distribution of funding_rounds values:")
    for v, c in val_counts.items():
        bar = "█" * min(int(c / val_counts.max() * 30), 30)
        rprint(f"    {int(v):>3} rounds: {c:>7,}  {bar}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    valid_rounds = converted.dropna()
    ax.hist(valid_rounds, bins=range(0, int(valid_rounds.max()) + 2),
            color="#e67e22", edgecolor="white", alpha=0.8, align="left")
    ax.set_title("Distribution of Funding Rounds", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Funding Rounds")
    ax.set_ylabel("Number of Companies")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    save_figure(fig, "10_funding_rounds.png")

# ===========================================================================
# SECTION 11 – COLUMN RECOMMENDATIONS
# ===========================================================================

def column_recommendations():
    section("SECTION 11 – COLUMN RECOMMENDATIONS")

    rprint("""
  ┌─────────────────────┬────────────────────┬────────────────────────────────────────────────┐
  │ Column              │ Recommendation      │ Reason                                         │
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────┤
  │ permalink           │ EXCLUDE             │ Unique identifier, no predictive signal        │
  │ name                │ EXCLUDE (as feature)│ Company name has no pattern; use as label only  │
  │ homepage_url        │ EXCLUDE             │ Raw URL; no direct signal for ML               │
  │ status              │ TARGET ONLY         │ Used to create the binary target label         │
  │ country_code        │ USE (encoded)       │ Geography matters; target-encode top countries  │
  │ state_code          │ CONDITIONAL         │ Only useful for USA companies                  │
  │ region              │ CONDITIONAL         │ Only useful for USA companies                  │
  │ city                │ EXCLUDE or group    │ Too high cardinality (10,000+ unique cities)   │
  │ category_list       │ TRANSFORM           │ Multi-hot encode top 30–50 categories          │
  │ funding_total_usd   │ USE with caution ⚠ │ Strong signal but POTENTIAL LEAKAGE (see §12)  │
  │ funding_rounds      │ USE with caution ⚠ │ Strong signal but POTENTIAL LEAKAGE (see §12)  │
  │ founded_at          │ USE (engineered)    │ Extract year; compute company age at prediction│
  │ first_funding_at    │ USE with caution ⚠ │ Time-to-first-funding is a valid early signal  │
  │ last_funding_at     │ HIGH LEAKAGE RISK ⚠│ Represents future activity; see §12            │
  └─────────────────────┴────────────────────┴────────────────────────────────────────────────┘
  """)

# ===========================================================================
# SECTION 12 – DATA LEAKAGE ANALYSIS
# ===========================================================================

def leakage_analysis():
    section("SECTION 12 – DATA LEAKAGE ANALYSIS")

    rprint("""
  DATA LEAKAGE means accidentally including information in your model that
  would NOT be available at the moment you are making a real prediction.
  This makes model accuracy look artificially high in testing, but the model
  fails completely when deployed in the real world.

  ┌────────────────────┬───────────────┬──────────────────────────────────────────────────┐
  │ Column             │ Leakage Risk  │ Explanation                                      │
  ├────────────────────┼───────────────┼──────────────────────────────────────────────────┤
  │ funding_total_usd  │ HIGH ⚠        │ Reflects TOTAL funding raised over the entire    │
  │                    │               │ life of the company. For a company that later     │
  │                    │               │ gets acquired, this includes acquisition-phase    │
  │                    │               │ funding raised AFTER the prediction point.        │
  │                    │               │ MITIGATION: Only use funding_total_usd up to a   │
  │                    │               │ fixed "prediction point" (e.g., 2 years after    │
  │                    │               │ founding). We don't have time-series funding      │
  │                    │               │ data here, so this column carries leakage risk.  │
  ├────────────────────┼───────────────┼──────────────────────────────────────────────────┤
  │ funding_rounds     │ HIGH ⚠        │ Same problem: total rounds includes rounds that  │
  │                    │               │ happen AFTER the outcome is determined.           │
  │                    │               │ A Series C round often occurs BECAUSE the company│
  │                    │               │ is doing well — it's a consequence of success,    │
  │                    │               │ not a predictor of it.                           │
  ├────────────────────┼───────────────┼──────────────────────────────────────────────────┤
  │ first_funding_at   │ LOW-MODERATE  │ The date of the FIRST funding event is a         │
  │                    │               │ relatively early signal. If our prediction point  │
  │                    │               │ is "after first funding", this is legitimate.     │
  │                    │               │ MITIGATION: Compute time_to_first_funding         │
  │                    │               │ (days from founding to first funding) as a        │
  │                    │               │ feature. This is available at prediction time.    │
  ├────────────────────┼───────────────┼──────────────────────────────────────────────────┤
  │ last_funding_at    │ VERY HIGH ⚠   │ This is the date of the MOST RECENT funding      │
  │                    │               │ round. For a company acquired in 2018, the last  │
  │                    │               │ funding might be from 2017 — just before         │
  │                    │               │ acquisition. Including this date essentially      │
  │                    │               │ tells the model "a deal is imminent."            │
  │                    │               │ → RECOMMEND: Exclude this column entirely.       │
  └────────────────────┴───────────────┴──────────────────────────────────────────────────┘

  SUMMARY:
  All funding-related columns carry some leakage risk because our dataset
  records TOTAL or FINAL values — not values as of a fixed early time point.
  In Phase 2 (preprocessing), we will define a strict "prediction point" and
  decide how to handle these columns defensibly.
  """)

# ===========================================================================
# SECTION 13 – PREDICTION-TIME DEFINITION
# ===========================================================================

def prediction_time_definition():
    section("SECTION 13 – DEFENSIBLE PREDICTION-TIME DEFINITION")

    rprint("""
  THE CORE QUESTION:
  "At what point in a company's life are we making a prediction?"

  Without answering this precisely, the model is meaningless — any feature
  that occurs AFTER this point is leakage.

  ─────────────────────────────────────────────────────────────────────────
  PROPOSED PREDICTION POINT
  ─────────────────────────────────────────────────────────────────────────

  We define the prediction point as:

      "12–24 months after the company's first funding round."

  RATIONALE:
  1. By the time a startup receives its first external funding, a real investor
     has already conducted due diligence. This is a natural "early stage"
     evaluation moment.

  2. "Time to first funding" (founded_at → first_funding_at) is legitimately
     available before the prediction point and captures momentum.

  3. This avoids using data that only exists at the END of a company's journey
     (like last_funding_at or total_funding across 10 years).

  ─────────────────────────────────────────────────────────────────────────
  FEATURES LEGITIMATELY AVAILABLE AT THE PREDICTION POINT
  ─────────────────────────────────────────────────────────────────────────

  ✅ founded_at → derived feature: founding_year, company_age_at_prediction
  ✅ first_funding_at → derived: time_to_first_funding (days)
  ✅ category_list → the startup's industry/sector at founding
  ✅ country_code → where the company is headquartered
  ✅ region / state_code → sub-national geography (for USA)
  ✅ funding_total_usd → ONLY if interpreted as "funding raised up to first round"
                         (we cannot isolate this perfectly from the current data)
  ✅ funding_rounds → ONLY the count as of first funding = 1 (trivially 1)

  ⚠  funding_total_usd → Use with documented caveat: this reflects lifetime total
  ⚠  funding_rounds → Same caveat
  ❌ last_funding_at → Exclude: reflects future funding activity
  ❌ status → This IS the target; never use as a feature
  ❌ permalink, name, homepage_url → identifiers, not signals

  ─────────────────────────────────────────────────────────────────────────
  RECOMMENDED ENGINEERED FEATURES (Phase 2)
  ─────────────────────────────────────────────────────────────────────────

  1. founding_year          = year extracted from founded_at
  2. time_to_first_funding  = (first_funding_at - founded_at).days
                              → how quickly did the startup attract funding?
  3. has_founding_date      = binary flag (1 if founded_at is not null)
  4. is_usa                 = binary flag (1 if country_code == 'USA')
  5. n_categories           = number of pipe-separated categories in category_list
  6. category_X             = multi-hot encoding for top 30 categories
  7. log_funding            = log10(funding_total_usd + 1)  [with leakage caveat]

  ─────────────────────────────────────────────────────────────────────────
  ACADEMIC CAVEAT
  ─────────────────────────────────────────────────────────────────────────

  This dataset does NOT provide time-series funding data per round.
  Therefore, perfect leakage elimination is not possible.
  The approach above MINIMISES leakage and must be clearly documented
  in the project report as a known limitation.
  """)

# ===========================================================================
# SECTION 14 – SAVE INTERMEDIATE DATA
# ===========================================================================

def save_intermediate_data(df_known, df_operating, funding_converted):
    section("SECTION 14 – SAVING INTERMEDIATE DATA")

    # Save labeled dataset (acquired + ipo = 1, closed = 0)
    labeled_path = os.path.join(DATA_DIR, "labeled_known_outcomes.csv")
    df_known.to_csv(labeled_path, index=False, encoding="utf-8")
    rprint(f"  Labeled dataset saved  : {labeled_path}  ({len(df_known):,} rows)")

    # Save operating companies separately (unknown outcome)
    operating_path = os.path.join(DATA_DIR, "operating_companies.csv")
    df_operating.to_csv(operating_path, index=False, encoding="utf-8")
    rprint(f"  Operating dataset saved: {operating_path}  ({len(df_operating):,} rows)")

    rprint("\n  NOTE: Original CSV was NEVER modified.")
    rprint("  These are derived files only, stored in /data.")

# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main():
    args = parse_args()

    rprint("=" * 70)
    rprint("  AI-Based Startup Viability Prediction and Validation System")
    rprint("  Phase 1 – Exploratory Data Analysis")
    rprint("=" * 70)
    rprint(f"  Dataset : {args.csv}")
    rprint(f"  Plots   : {PLOTS_DIR}")
    rprint(f"  Reports : {REPORTS_DIR}")
    rprint(f"  Data    : {DATA_DIR}")

    # Run all analysis sections in order
    df                        = load_data(args.csv)
    df                        = basic_info(df)
    df_known, df_operating    = analyze_status(df)
    missing_values(df, df_known)
    duplicates(df)
    funding_converted         = inspect_funding(df)
    inspect_dates(df)
    analyze_categories(df)
    analyze_geography(df)
    inspect_funding_rounds(df)
    column_recommendations()
    leakage_analysis()
    prediction_time_definition()
    save_intermediate_data(df_known, df_operating, funding_converted)

    # -----------------------------------------------------------------------
    # Write the full report to file
    # -----------------------------------------------------------------------
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  [REPORT SAVED] {REPORT_PATH}")

    section("EDA COMPLETE")
    rprint("  All outputs saved. Original CSV was NOT modified.")
    rprint("  Next step: Review outputs/, then proceed to Phase 2 preprocessing.")

if __name__ == "__main__":
    main()
