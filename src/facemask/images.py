"""Locating image files on disk.

Kept free of TensorFlow imports so that file-level operations -- listing a
directory, splitting a dataset -- do not pay the several-second cost of loading
the framework.
"""

from __future__ import annotations

from pathlib import Path

#: File extensions treated as images, matching what imutils.paths.list_images
#: accepted in the previous implementation.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def find_images(directory: Path) -> list[Path]:
    """Recursively collect image files beneath a directory, in a stable order.

    Replaces ``imutils.paths.list_images``. Sorting makes dataset order
    reproducible across platforms, which the previous implementation did not
    guarantee because it depended on filesystem walk order.

    Args:
        directory: Directory to search.

    Returns:
        Sorted paths of every image file found beneath ``directory``.
    """
    return sorted(
        path
        for path in Path(directory).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
