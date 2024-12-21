import toml
import subprocess

def generate_pyproject_with_env(output_path="pyproject.toml"):
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        dependencies = {}
        for line in result.stdout.splitlines():
            if "==" in line:
                package, version = line.split("==")
                dependencies[package] = f"^{version}"

        pyproject_content = {
            "tool": {
                "poetry": {
                    "name": "my-project",
                    "version": "0.1.0",
                    "description": "A sample Python project",
                    "authors": ["Your Name <your.email@example.com>"],
                    "dependencies": {
                        "python": "^3.8",
                        **dependencies
                    },
                }
            },
            "build-system": {
                "requires": ["poetry-core>=1.0.0"],
                "build-backend": "poetry.core.masonry.api",
            }
        }

        # write pyproject.toml 
        with open(output_path, "w") as f:
            toml.dump(pyproject_content, f)
        print(f"'pyproject.toml' in {output_path}")

    except subprocess.CalledProcessError as e:
        print("Generate pyproject.toml fail:", e)

generate_pyproject_with_env()
