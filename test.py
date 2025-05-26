import os
import shutil
from pathlib import Path
from core.analyzer import CodebaseAnalyzer

def create_sample_code_files(base_dir: Path):
    # Create some sample code files in base_dir
    files = {
        'hello.py': "def greet():\n    print('Hello, World!')\n",
        'math_utils.py': (
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
        ),
        'ignored.txt': "This should be ignored based on extension\n"
    }

    for filename, content in files.items():
        file_path = base_dir / filename
        file_path.write_text(content)

def test_codebase_analyzer():
    # Setup test directory
    test_dir = Path('test_project')  # Change to your desired directory path
    # if test_dir.exists():
    #     shutil.rmtree(test_dir)
    # test_dir.mkdir()

    # create_sample_code_files(test_dir)

    analyzer = CodebaseAnalyzer()

    print("=== Analyzing directory ===")
    results = analyzer.analyze_path(test_dir, recursive=False)
    print(f"Files processed: {results['files_processed']}")
    print(f"Chunks extracted: {results['chunks_extracted']}")
    print(f"Embeddings generated: {results['embeddings_generated']}")
    print(f"Files skipped: {results['files_skipped']}")
    print(f"Errors: {results['errors']}")
    print(f"Changes summary: {results['changes']}")
    print(f"Merkle root hash: {results['merkle_hash']}")

    # Analyze single file
    single_file = test_dir / 'hello.py'
    print("\n=== Analyzing single file ===")
    single_file_results = analyzer.analyze_single_file(single_file)
    print(single_file_results)

    # Cleanup
    # shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_codebase_analyzer()
