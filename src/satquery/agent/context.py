"""Per-run state shared between the controller and its tools.

Tools never talk to each other directly: a specialist writes into ``artifacts`` and
the VLM step reads from it. That is what turns a deterministic measurement into
grounded evidence inside the prompt, and it keeps every hand-off visible in the
trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from satquery.eval.backends.base import VLMBackend
from satquery.geo.raster import RasterInfo
from satquery.schema import ImageRef, ImageRole, InputConfig


@dataclass
class RunContext:
    """Everything a tool may read or write during one query."""

    run_id: str
    query: str
    images: tuple[ImageRef, ...]
    infos: tuple[RasterInfo, ...]
    input_config: InputConfig
    workdir: Path
    backend: VLMBackend
    artifacts: dict[str, Any] = field(default_factory=dict)

    def image_by_role(self, role: ImageRole) -> ImageRef | None:
        for image in self.images:
            if image.role is role:
                return image
        return None

    def require_role(self, role: ImageRole) -> ImageRef:
        image = self.image_by_role(role)
        if image is None:
            available = ", ".join(str(i.role) for i in self.images)
            raise ValueError(
                f"no image with role '{role}'; available roles: {available}"
            )
        return image

    def info_for(self, image: ImageRef) -> RasterInfo | None:
        for info in self.infos:
            if info.path == image.path:
                return info
        return None

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(image.path for image in self.images)

    def artifact_path(self, filename: str) -> Path:
        """Absolute path for a tool-produced file, directory created on demand."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        return self.workdir / filename

    def artifact_uri(self, filename: str) -> str:
        """Stable URI the API serves evidence from."""
        return f"artifacts/{self.run_id}/{filename}"
