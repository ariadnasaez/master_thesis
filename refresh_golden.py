"""
Refresh `expected_rows` in golden_dataset.json by re-executing every golden_query
against the current state of `tfm_datanex`.

Use this when the database has been updated since the golden dataset was created
and the stale expected_rows are causing false negatives in evaluate.py.

Usage:
    python refresh_golden.py                # writes golden_dataset_refreshed.json (safe)
    python refresh_golden.py --in-place     # overwrites golden_dataset.json directly
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession


HERE = Path(__file__).parent
GOLDEN_PATH = HERE / "golden_dataset.json"
REFRESHED_PATH = HERE / "golden_dataset_refreshed.json"


async def refresh(in_place: bool):
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    server_params = StdioServerParameters(command=sys.executable, args=[str(HERE / "server.py")])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            unchanged = 0
            updated = 0
            failed = []

            for item in golden:
                qid = item.get("id")
                query = item.get("golden_query", "").strip()
                if not query:
                    continue

                try:
                    result = await session.call_tool("execute_query", {"query": query})
                    text = result.content[0].text
                    new_rows_dicts = json.loads(text) if text else []
                    # Convert dict rows back to list rows (the format used in expected_rows)
                    new_rows = []
                    for row in new_rows_dicts:
                        if isinstance(row, dict):
                            new_rows.append(list(row.values()))
                        else:
                            new_rows.append(row)
                except Exception as e:
                    failed.append((qid, str(e)))
                    print(f"[{qid:>3}] ❌ FAILED: {e}")
                    continue

                old_rows = item.get("expected_rows", [])
                if old_rows == new_rows:
                    unchanged += 1
                    print(f"[{qid:>3}] ✓  unchanged ({len(new_rows)} rows)")
                else:
                    updated += 1
                    print(f"[{qid:>3}] ⟳  updated  ({len(old_rows)} → {len(new_rows)} rows)")
                    item["expected_rows"] = new_rows

    out_path = GOLDEN_PATH if in_place else REFRESHED_PATH
    with open(out_path, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"Unchanged: {unchanged}")
    print(f"Updated:   {updated}")
    print(f"Failed:    {len(failed)}")
    if failed:
        for qid, err in failed:
            print(f"  - id {qid}: {err[:80]}")
    print(f"\n→ {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite golden_dataset.json directly. By default writes to golden_dataset_refreshed.json.")
    args = parser.parse_args()
    asyncio.run(refresh(args.in_place))
