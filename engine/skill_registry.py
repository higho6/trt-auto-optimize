"""Discover local skill bundles without depending on Claude's skill loader."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    directory: Path
    source: str
    instructions: str
    sha256: str

    def descriptor(self) -> dict[str, str]:
        return {
            "name": self.name, "description": self.description,
            "directory": str(self.directory), "source": self.source,
            "sha256": self.sha256,
        }


class SkillRegistry:
    def __init__(self, roots: list[tuple[Path, str]]) -> None:
        self.skills: dict[str, Skill] = {}
        for root, source in roots:
            if not root.is_dir():
                raise FileNotFoundError(f"Skill directory missing: {root}")
            for path in sorted(root.rglob("SKILL.md")):
                path = path.resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError(f"Skill escapes its source root: {path}")
                content = path.read_text()
                match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", content, re.DOTALL)
                if not match:
                    raise ValueError(f"Missing YAML frontmatter: {path}")
                metadata = yaml.safe_load(match.group(1))
                if not isinstance(metadata, dict):
                    raise ValueError(f"Invalid skill frontmatter: {path}")
                name, description = metadata.get("name"), metadata.get("description")
                if (
                    not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,64}", name)
                    or name != path.parent.name
                    or not isinstance(description, str) or not description.strip()
                ):
                    raise ValueError(f"Invalid skill name/directory/description: {path}")
                if name in self.skills:
                    raise ValueError(f"Duplicate skill name: {name}")
                self.skills[name] = Skill(
                    name, description, path.parent, source, content[match.end():],
                    hashlib.sha256(content.encode()).hexdigest(),
                )
        if not self.skills:
            raise ValueError("No SKILL.md files discovered; initialize the KDA submodules")

    def get(self, name: str) -> Skill:
        if name not in self.skills:
            raise ValueError(f"Unknown skill: {name}")
        return self.skills[name]

    def catalog(self) -> list[dict[str, str]]:
        return [skill.descriptor() for skill in self.skills.values()]

    def resource(self, name: str, relative_path: str) -> Path:
        skill = self.get(name)
        path = (skill.directory / relative_path).resolve()
        if not path.is_relative_to(skill.directory):
            raise ValueError(f"Skill resource escapes its bundle: {relative_path}")
        return path

    def list_resources(self, name: str, relative_path: str) -> list[dict[str, str]]:
        directory = self.resource(name, relative_path)
        if not directory.is_dir():
            raise ValueError(f"Not a skill resource directory: {relative_path}")
        return [
            {"path": str(path.relative_to(self.get(name).directory)),
             "kind": "directory" if path.is_dir() else "file"}
            for path in sorted(directory.iterdir()) if not path.name.startswith(".")
        ]
