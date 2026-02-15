"""
Migrate IRS search data from CSV to PostgreSQL.

Usage:
  python migrate_irs.py                          # uses default local connection
  python migrate_irs.py postgresql://user:pass@host:port/dbname  # Railway or custom
"""

import csv
import sys
import psycopg2

CSV_PATH = "EVENTLEADS/irs_search.csv"

# The 40 real columns (everything before the trailing empty columns)
COLUMNS = [
    "EIN", "OrganizationName", "Website", "PhysicalAddress", "PhysicalCity",
    "PhysicalState", "PhysicalZIP", "BusinessOfficerPhone", "PrincipalOfficerName",
    "TotalRevenue", "GrossReceipts", "NetIncome", "TotalAssets", "ContributionsReceived",
    "ProgramServiceRevenue", "FundraisingGrossIncome", "FundraisingDirectExpenses",
    "Event1Name", "Event1GrossReceipts", "Event1GrossRevenue", "Event1NetIncome",
    "Event2Name", "Event2GrossReceipts", "Event2GrossRevenue",
    "Event1Keyword", "Event2Keyword", "PrimaryEventType", "ProspectTier", "ProspectScore",
    "Region5", "MissionDescriptionShort",
    "HasGala", "HasAuction", "HasRaffle", "HasBall", "HasDinner",
    "HasBenefit", "HasTournament", "HasGolf", "HasFundraisingActivities",
]

CREATE_TABLE = """
DROP TABLE IF EXISTS tax_year_2019_search;
CREATE TABLE tax_year_2019_search (
    EIN TEXT,
    OrganizationName TEXT,
    Website TEXT,
    PhysicalAddress TEXT,
    PhysicalCity TEXT,
    PhysicalState TEXT,
    PhysicalZIP TEXT,
    BusinessOfficerPhone TEXT,
    PrincipalOfficerName TEXT,
    TotalRevenue BIGINT,
    GrossReceipts BIGINT,
    NetIncome BIGINT,
    TotalAssets BIGINT,
    ContributionsReceived BIGINT,
    ProgramServiceRevenue BIGINT,
    FundraisingGrossIncome BIGINT,
    FundraisingDirectExpenses BIGINT,
    Event1Name TEXT,
    Event1GrossReceipts BIGINT,
    Event1GrossRevenue BIGINT,
    Event1NetIncome BIGINT,
    Event2Name TEXT,
    Event2GrossReceipts BIGINT,
    Event2GrossRevenue BIGINT,
    Event1Keyword TEXT,
    Event2Keyword TEXT,
    PrimaryEventType TEXT,
    ProspectTier TEXT,
    ProspectScore NUMERIC,
    Region5 TEXT,
    MissionDescriptionShort TEXT,
    HasGala INTEGER,
    HasAuction INTEGER,
    HasRaffle INTEGER,
    HasBall INTEGER,
    HasDinner INTEGER,
    HasBenefit INTEGER,
    HasTournament INTEGER,
    HasGolf INTEGER,
    HasFundraisingActivities INTEGER
);
"""

BIGINT_COLS = {
    "TotalRevenue", "GrossReceipts", "NetIncome", "TotalAssets", "ContributionsReceived",
    "ProgramServiceRevenue", "FundraisingGrossIncome", "FundraisingDirectExpenses",
    "Event1GrossReceipts", "Event1GrossRevenue", "Event1NetIncome",
    "Event2GrossReceipts", "Event2GrossRevenue",
}

INT_COLS = {
    "HasGala", "HasAuction", "HasRaffle", "HasBall", "HasDinner",
    "HasBenefit", "HasTournament", "HasGolf", "HasFundraisingActivities",
}

NUMERIC_COLS = {"ProspectScore"}


def clean_value(col, val):
    """Convert NULL strings and empty strings to None, cast numeric types."""
    if val is None or val.strip() == "" or val.strip().upper() == "NULL":
        return None
    val = val.strip()
    if col in BIGINT_COLS or col in INT_COLS:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None
    if col in NUMERIC_COLS:
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return val


def main():
    conn_string = sys.argv[1] if len(sys.argv) > 1 else "postgresql://localhost/irs"
    print(f"Connecting to: {conn_string}")
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()

    print("Creating table...")
    cur.execute(CREATE_TABLE)
    conn.commit()

    placeholders = ", ".join(["%s"] * len(COLUMNS))
    col_names = ", ".join(COLUMNS)
    insert_sql = f"INSERT INTO tax_year_2019_search ({col_names}) VALUES ({placeholders})"

    print(f"Importing from {CSV_PATH}...")
    batch = []
    batch_size = 1000
    total = 0

    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = tuple(clean_value(col, row.get(col, "")) for col in COLUMNS)
            batch.append(values)
            if len(batch) >= batch_size:
                cur.executemany(insert_sql, batch)
                conn.commit()
                total += len(batch)
                print(f"  {total:,} rows imported...", end="\r")
                batch = []

    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()
        total += len(batch)

    print(f"\nDone! {total:,} rows imported.")

    # Create indexes for common query patterns
    print("Creating indexes...")
    cur.execute("CREATE INDEX idx_irs_state ON tax_year_2019_search (PhysicalState);")
    cur.execute("CREATE INDEX idx_irs_revenue ON tax_year_2019_search (TotalRevenue DESC);")
    cur.execute("CREATE INDEX idx_irs_orgname ON tax_year_2019_search (OrganizationName);")
    cur.execute("CREATE INDEX idx_irs_region ON tax_year_2019_search (Region5);")
    cur.execute("CREATE INDEX idx_irs_event_type ON tax_year_2019_search (PrimaryEventType);")
    cur.execute("CREATE INDEX idx_irs_prospect ON tax_year_2019_search (ProspectTier);")
    conn.commit()
    print("Indexes created.")

    cur.close()
    conn.close()
    print("Migration complete!")


if __name__ == "__main__":
    main()
