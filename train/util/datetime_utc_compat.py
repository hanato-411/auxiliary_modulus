"""Python 3.10 向け: stdlib の datetime に UTC エイリアスを付与する（3.11 の datetime.UTC と同等）。"""

from __future__ import annotations

import datetime

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined, assignment]
