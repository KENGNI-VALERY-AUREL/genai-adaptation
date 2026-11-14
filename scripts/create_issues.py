"""Remove duplicate issues (keep the lowest number per title) and close finished tasks.

Usage (from the repository root):
    python scripts/cleanup_issues.py --dry-run   # preview only
    python scripts/cleanup_issues.py             # apply
"""
import json
import subprocess
import sys

DRY_RUN = "--dry-run" in sys.argv
# Tasks already completed: their issues must be closed
DONE = {"A1", "A3", "A4", "A5", "B1", "B2", "C1", "C2", "C3",
        "D1", "D2", "E1", "F2", "F3"}


def gh(args, check=True):
    """Run a gh command and return its output."""
    result = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if check and result.returncode != 0:
        print("  failed:", result.stderr.strip())
    return result.stdout


def main():
    out = gh(["issue", "list", "--state", "all", "--limit", "200",
              "--json", "number,title,state"])
    issues = json.loads(out)

    # Group issues by title, then sort each group by number
    groups = {}
    for issue in issues:
        groups.setdefault(issue["title"], []).append(issue)

    for title, group in groups.items():
        group.sort(key=lambda i: i["number"])
        keep, duplicates = group[0], group[1:]

        # 1) Delete duplicates (fall back to closing if deletion is not allowed)
        for dup in duplicates:
            print(f"duplicate #{dup['number']} of #{keep['number']}: {title}")
            if DRY_RUN:
                continue
            res = subprocess.run(
                ["gh", "issue", "delete", str(dup["number"]), "--yes"],
                capture_output=True, text=True)
            if res.returncode != 0:
                print("  delete failed, closing instead")
                gh(["issue", "close", str(dup["number"]),
                    "--reason", "not planned"])

        # 2) Close the kept issue if its task is already done
        task_id = title.split("]")[0].lstrip("[")
        if task_id in DONE and keep["state"] == "OPEN":
            print(f"closing #{keep['number']}: {title}")
            if not DRY_RUN:
                gh(["issue", "close", str(keep["number"]),
                    "--reason", "completed"])

    print("Done.")


if __name__ == "__main__":
    main()