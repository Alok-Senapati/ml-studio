from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEMPLATE_DIR = PROJECT_ROOT / "templates" / "ml_template"
PROJECTS_DIR = PROJECT_ROOT / "projects"

PLACEHOLDERS = {
    "__PROJECT_NAME__": "project_name",
    "__PROJECT_DESCRIPTION__": "project_description",
    "__AUTHOR__": "author",
    "__YEAR__": "year",
}