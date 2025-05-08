from pathlib import Path
import os
import argparse
from collections import defaultdict
import re
from multiprocessing import Pool, cpu_count
from typing import Tuple, List
import logging

try:
    import pymupdf
    PDF_TOOLCHAIN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PDF_TOOLCHAIN_AVAILABLE = False

    class NotImported:
        def __getattr__(self, item):
            raise ModuleNotFoundError(
                "PyMuPDF dependencies are not installed. "
                "Please install them using: pip install 'rag-agent[loaders]'"
            )

        def __call__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "PyMuPDF dependencies are not installed. "
                "Please install them using: pip install 'rag-agent[loaders]'"
            )

    globals().update(dict.fromkeys(
        [
            "pymupdf",
        ],
        NotImported()
    ))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("rag_agent.utils.merge")


def natural_sort_key(p: Path):
    """
    Sort strings containing numbers in a natural way.
    For example: ["1.pdf", "2.pdf", "10.pdf"] instead of ["1.pdf", "10.pdf", "2.pdf"]
    """
    return [
        int(t) if t.isdigit() else t for t in re.split(r'(\d+)', p.name) if t
    ]


def merge_pdf_group(args: Tuple[List[Path], Path]) -> None:
    """
    Merge a group of PDF files into a single output file.
    This function is designed to be run in a separate process.

    Args:
        args: Tuple containing (list of PDF files to merge, output file path)
    """
    files, output_file = args
    logger.info(f"Processing {len(files)} files into {output_file}")

    merged_pdf = pymupdf.open()
    for pdf_file in sorted(files, key=natural_sort_key):
        logger.debug(f"Adding: {pdf_file.name}")
        doc = pymupdf.open(str(pdf_file))
        merged_pdf.insert_pdf(doc)
        doc.close()

    merged_pdf.save(str(output_file))
    merged_pdf.close()
    logger.info(f"Created merged PDF: {output_file}")


def merge_pdfs_in_directory(
    directory_path: str = os.getcwd(),
    glob_pattern: str = "*.pdf",
    output_dir: str = None,
    num_processes: int = None,
    prefix: str = "merged_"
):
    """
    Merge PDF files in the given directory and its subdirectories.
    Files are grouped by their parent directory and merged into a single PDF.

    Args:
        directory_path: Path to the directory containing PDF files (default: current working directory)
        glob_pattern: Pattern to match PDF files (default: "*.pdf")
        output_dir: Directory to save merged PDFs (default: same as input directory)
        num_processes: Number of processes to use (default: number of CPU cores)
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = directory

    pdf_files = list(directory.rglob(glob_pattern))
    if not pdf_files:
        logger.warning(f"No PDF files found matching pattern '{glob_pattern}' in {directory_path}")
        return

    files_by_directory = defaultdict(list)
    for pdf_file in pdf_files:
        files_by_directory[pdf_file.parent].append(pdf_file)

    merge_tasks = []
    for parent_dir, files in files_by_directory.items():
        rel_path = parent_dir.relative_to(directory)
        output_name = f"{prefix}{str(rel_path).replace(os.sep, '_')}.pdf"
        output_file = output_dir / output_name
        merge_tasks.append((files, output_file))

    num_processes = num_processes or max(1, cpu_count() - 1)
    logger.info(f"Using {num_processes} processes for merging")

    with Pool(processes=num_processes) as pool:
        pool.map(merge_pdf_group, merge_tasks)


def main():
    parser = argparse.ArgumentParser(
        prog="rag_agent.utils.merge",
        description="Merge PDF files in a directory and its subdirectories",
        add_help=True
    )
    parser.add_argument("directory", help="Path to the directory containing PDF files", required=True)
    parser.add_argument("--pattern", default="*.pdf", help="Glob pattern to match PDF files (default: *.pdf)")
    parser.add_argument("--output-dir", help="Directory to save merged PDFs (default: same as input directory)")
    parser.add_argument("--processes", type=int, help="Number of processes to use (default: number of CPU cores - 1)")
    parser.add_argument("--prefix", default="merged_", help="Prefix for merged PDF files (default: merged_)")

    args = parser.parse_args()
    try:
        merge_pdfs_in_directory(
            directory_path=args.directory,
            glob_pattern=args.pattern,
            output_dir=args.output_dir,
            num_processes=args.processes,
            prefix=args.prefix
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
