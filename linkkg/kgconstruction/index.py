import argparse
import importlib
import pkgutil
import shutil
import subprocess
import sys
from pathlib import Path

from graphrag.config.enums import IndexingMethod
try:
    from graphrag.index.cli import index_cli as _index_cli
except ModuleNotFoundError:
    try:
        from graphrag.cli.index import index_cli as _index_cli
    except ModuleNotFoundError:
        _index_cli = None

from monkey_patch import patch_openai_embeddings_llm

patch_openai_embeddings_llm()


def _run_index(
    root_dir: Path,
    *,
    verbose: bool,
    cache: bool,
    dry_run: bool,
    skip_validation: bool,
) -> None:
    def _register_file_reader() -> None:
        try:
            from graphrag_input.input_reader_factory import register_input_reader
            from graphrag_input.text import TextFileReader
        except ModuleNotFoundError:
            return

        for reader_type in ("file", "text"):
            try:
                register_input_reader(reader_type, TextFileReader, scope="transient")
            except Exception:
                continue

    def _import_all_submodules(package_name: str) -> None:
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError:
            return

        if not hasattr(package, "__path__"):
            return

        for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            try:
                importlib.import_module(module_info.name)
            except Exception:
                continue

    if _index_cli is not None:
        # Ensure local plugins are imported so reader/cache registration occurs.
        for package_name in (
            "graphrag_input",
            "graphrag_cache",
            "graphrag_storage",
            "graphrag_vectors",
        ):
            _import_all_submodules(package_name)
        _index_cli(
            root_dir=root_dir,
            method=IndexingMethod.Standard,
            verbose=verbose,
            cache=cache,
            dry_run=dry_run,
            skip_validation=skip_validation,
        )
        return

    graphrag_cli = shutil.which("graphrag")
    if not graphrag_cli:
        raise RuntimeError("graphrag CLI not found in PATH")
    cmd = [
        graphrag_cli,
        "index",
        "--root",
        str(root_dir),
        "--method",
        "standard",
    ]
    if verbose:
        cmd.append("--verbose")
    if cache:
        cmd.append("--cache")
    else:
        cmd.append("--no-cache")
    if dry_run:
        cmd.append("--dry-run")
    if skip_validation:
        cmd.append("--skip-validation")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--skip-validations", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--nocache", action="store_true")
    args = parser.parse_args()

    use_cache = args.cache and not args.nocache

    _run_index(
        Path(args.root),
        verbose=args.verbose or False,
        cache=use_cache,
        dry_run=args.dryrun or False,
        skip_validation=args.skip_validations or False,
    )


if __name__ == "__main__":
    main()