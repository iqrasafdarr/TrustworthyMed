import os
from itertools import combinations
from PIL import Image
import imagehash


class PHashDuplicateAuditor:
    """
    Finds near-duplicate images (same lesion photographed twice, cropped,
    rotated, etc.) using perceptual hashing. Catches leakage that a plain
    lesion_id check would miss.
    """

    def __init__(self, image_dir, hash_size=8, distance_threshold=5):
        self.image_dir = image_dir
        self.hash_size = hash_size
        self.distance_threshold = distance_threshold  # lower = stricter match

    def compute_hashes(self, image_filenames):
        """Returns {filename: phash} for every image."""
        hashes = {}
        for fname in image_filenames:
            path = os.path.join(self.image_dir, fname)
            try:
                img = Image.open(path)
                hashes[fname] = imagehash.phash(img, hash_size=self.hash_size)
            except Exception as e:
                print(f"Skipping {fname}: {e}")
        return hashes

    def find_duplicates(self, image_filenames):
        """
        Compares every image against every other image and flags pairs
        whose hashes are close enough to be considered near-duplicates.
        """
        hashes = self.compute_hashes(image_filenames)
        duplicates = []

        for (name_a, hash_a), (name_b, hash_b) in combinations(hashes.items(), 2):
            distance = hash_a - hash_b  # Hamming distance between hashes
            if distance <= self.distance_threshold:
                duplicates.append((name_a, name_b, distance))

        return duplicates

    def report(self, image_filenames):
        dupes = self.find_duplicates(image_filenames)
        if dupes:
            print(f"Found {len(dupes)} near-duplicate pairs:")
            for a, b, d in dupes:
                print(f"  {a} <-> {b} (distance={d})")
        else:
            print("No near-duplicates found.")
        return dupes