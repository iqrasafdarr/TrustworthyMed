import pandas as pd


class SplitVerifier:
    """
    Checks that no lesion/patient appears in both the train and test splits.
    Prevents data leakage that would make results look better than they really are.
    """

    def __init__(self, metadata_path):
        self.metadata = pd.read_csv(metadata_path)

    def check_leakage(self, train_ids, test_ids, id_column="lesion_id"):
        """
        train_ids, test_ids: lists/sets of image_id values in each split
        id_column: the column that groups images belonging to the same lesion/patient
        """
        train_set = set(train_ids)
        test_set = set(test_ids)

        # Map each image_id to its lesion_id
        id_to_group = dict(zip(self.metadata["image_id"], self.metadata[id_column]))

        train_groups = {id_to_group[i] for i in train_set if i in id_to_group}
        test_groups = {id_to_group[i] for i in test_set if i in id_to_group}

        overlapping_groups = train_groups & test_groups

        leaked_train_images = [
            img for img in train_set
            if id_to_group.get(img) in overlapping_groups
        ]
        leaked_test_images = [
            img for img in test_set
            if id_to_group.get(img) in overlapping_groups
        ]

        return {
            "has_leakage": len(overlapping_groups) > 0,
            "overlapping_lesion_count": len(overlapping_groups),
            "leaked_train_images": leaked_train_images,
            "leaked_test_images": leaked_test_images,
        }

    def report(self, train_ids, test_ids, id_column="lesion_id"):
        result = self.check_leakage(train_ids, test_ids, id_column)
        if result["has_leakage"]:
            print(f"LEAKAGE FOUND: {result['overlapping_lesion_count']} lesions appear in both splits")
            print(f"  Affected train images: {len(result['leaked_train_images'])}")
            print(f"  Affected test images: {len(result['leaked_test_images'])}")
        else:
            print("No leakage detected — train and test splits are clean.")
        return result