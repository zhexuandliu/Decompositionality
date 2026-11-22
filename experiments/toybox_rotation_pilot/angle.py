"""Approximate in-plane rotation between two recorded frames, from SIFT matches + a RANSAC similarity transform.
Used only as a CHECK on visually selected frames (hands and slight out-of-plane motion make it approximate)."""
from __future__ import annotations

import math

import cv2
import numpy as np

_sift = cv2.SIFT_create(nfeatures=4000)


def _gray(img, scale=0.5):
    a = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    return cv2.resize(a, None, fx=scale, fy=scale)


def rotation_deg(img_a, img_b):
    """Signed rotation (degrees, positive = counter-clockwise as seen in the image) taking A to B,
    plus the number of RANSAC inliers. Returns (None, 0) if no reliable estimate."""
    ga, gb = _gray(img_a), _gray(img_b)
    ka, da = _sift.detectAndCompute(ga, None)
    kb, db = _sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return None, 0
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return None, 0
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None:
        return None, 0
    # image y axis points down, so a counter-clockwise rotation on screen has a negative matrix angle
    ang = -math.degrees(math.atan2(M[1, 0], M[0, 0]))
    return ang, int(inl.sum())
