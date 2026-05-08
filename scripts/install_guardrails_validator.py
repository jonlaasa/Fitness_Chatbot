from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


DEFAULT_VENDOR_PATH = r"C:\gr_pkgs"


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def main() -> None:
    vendor_path = Path(os.getenv("GUARDRAILS_VENDOR_PATH", DEFAULT_VENDOR_PATH))
    vendor_path.mkdir(parents=True, exist_ok=True)

    python_executable = Path(sys.executable)
    print(f"Installing Guardrails dependencies into: {vendor_path}")

    _run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--target",
            str(vendor_path),
            "guardrails-ai",
        ]
    )
    _run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--target",
            str(vendor_path),
            "git+https://github.com/tryolabs/restricttotopic.git",
        ]
    )

    print("Guardrails validator installation completed.")


if __name__ == "__main__":
    main()
