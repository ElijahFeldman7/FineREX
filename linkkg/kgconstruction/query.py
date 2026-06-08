import sys
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--method", required=True, type=str)
    parser.add_argument("query", nargs="+")

    args, unknown = parser.parse_known_args()

    query_str = " ".join(args.query)

    cmd = [
        sys.executable,
        "-m",
        "graphrag",
        "query",
        "--root",
        args.root,
        "--method",
        args.method,
        query_str,
    ]

    print(f"Running Query via Subprocess: {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd).returncode)