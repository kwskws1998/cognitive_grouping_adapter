"""Check syntax, source style and input configurations without writing bytecode."""

import ast
import io
import tokenize

from _common import PROJECT_ROOT, read_config
from cog_grouping.provenance import atomic_json, source_fingerprint


def main():
    """Audit owned code only; preserve unmodified upstream code and license headers."""
    checked = []
    for folder in ("src", "scripts", "tests"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
            content = path.read_text(encoding="utf-8")
            ast.parse(content, filename=str(path))
            compile(content, str(path), "exec")
            comments = [token for token in tokenize.generate_tokens(io.StringIO(content).readline)
                        if token.type == tokenize.COMMENT]
            if comments:
                raise ValueError("owned code uses inline comments: " + str(path))
            checked.append(str(path.relative_to(PROJECT_ROOT)))
    configurations = []
    for path in sorted((PROJECT_ROOT / "configs").glob("*.json")):
        read_config(path)
        configurations.append(str(path.relative_to(PROJECT_ROOT)))
    report = {"syntax_ok": True, "inline_comment_count": 0, "python_files": checked,
              "valid_configs": configurations, "source": source_fingerprint(PROJECT_ROOT)}
    atomic_json(PROJECT_ROOT / "reports" / "source_audit.json", report)
    print("Syntax/style/configuration audit passed for %d Python files and %d configs" % (len(checked), len(configurations)))


if __name__ == "__main__":
    main()
