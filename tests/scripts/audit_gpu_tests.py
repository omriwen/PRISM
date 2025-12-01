"""Audit GPU test coverage in the test suite."""

import ast
from pathlib import Path


def find_gpu_capable_tests(test_dir: Path) -> dict[str, list[str]]:
    """Find all tests that use PyTorch GPU operations."""
    results = {}

    for test_file in test_dir.rglob("test_*.py"):
        with open(test_file) as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue

        gpu_tests = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                source = ast.unparse(node)

                # GPU indicators
                if any(indicator in source for indicator in [
                    "torch.cuda",
                    ".to(device)",
                    ".cuda()",
                    "gpu_device",
                    "ProgressiveDecoder",
                    "PRISMTrainer",
                    "nn.Module",
                    "requires_grad",
                ]):
                    # Check if already has GPU marker
                    has_marker = any(
                        "pytest.mark.gpu" in ast.unparse(dec)
                        for dec in node.decorator_list
                    )

                    if not has_marker:
                        gpu_tests.append((node.name, node.lineno, "NO_MARKER"))
                    else:
                        gpu_tests.append((node.name, node.lineno, "HAS_MARKER"))

        if gpu_tests:
            results[str(test_file.relative_to(test_dir))] = gpu_tests

    return results


if __name__ == "__main__":
    from pprint import pprint

    results = find_gpu_capable_tests(Path("tests"))

    print("=== GPU Test Coverage Audit ===\n")

    total_gpu_tests = 0
    missing_markers = 0

    for file_path, tests in sorted(results.items()):
        print(f"\n{file_path}:")
        for test_name, lineno, status in tests:
            total_gpu_tests += 1
            if status == "NO_MARKER":
                missing_markers += 1
                print(f"  ❌ {test_name}:{lineno} - Missing @pytest.mark.gpu")
            else:
                print(f"  ✅ {test_name}:{lineno}")

    print(f"\n=== Summary ===")
    print(f"Total GPU-capable tests: {total_gpu_tests}")
    print(f"Missing markers: {missing_markers}")
    print(f"Coverage: {(total_gpu_tests - missing_markers) / total_gpu_tests * 100:.1f}%")
