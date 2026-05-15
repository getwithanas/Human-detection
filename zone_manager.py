"""
zone_manager.py — Manages restricted polygonal zones.
Uses OpenCV's pointPolygonTest for accurate inside/outside checks.
"""

import cv2
import numpy as np
from typing import List, Tuple

Polygon = List[Tuple[int, int]]


class ZoneManager:
    """
    Stores alert zones as contours and tests whether a point
    (person centre) lies inside any of them.
    """

    def __init__(self, zone_polygons: List[Polygon]):
        self.zone_polygons = zone_polygons
        self._contours = [
            np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            for poly in zone_polygons
        ]

    def check(self, point: Tuple[int, int]) -> Tuple[bool, str]:
        """
        Returns (inside: bool, zone_name: str).
        zone_name is empty string if not inside any zone.
        """
        for i, contour in enumerate(self._contours):
            dist = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
            if dist >= 0:                     # 0 = on edge, positive = inside
                name = f"Zone {chr(65 + i)}" # Zone A, B, C …
                return True, name
        return False, ""

    def add_zone(self, polygon: Polygon) -> None:
        self.zone_polygons.append(polygon)
        self._contours.append(
            np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        )

    def clear(self) -> None:
        self.zone_polygons.clear()
        self._contours.clear()
