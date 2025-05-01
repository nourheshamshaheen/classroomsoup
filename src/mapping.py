import os
import nltk
from nltk.corpus import wordnet as wn

# Ensure the NLTK WordNet data is downloaded.
nltk.download("wordnet")


##############################################################
# Helper function: Compute ImageNet fine-to-coarse mapping
# using the WordNet hierarchy and the class index JSON file.
##############################################################
def get_coarse_class(wnid, depth_threshold=5):
    """
    For a given WordNet ID (e.g., 'n01440764'), traverse up the hypernym chain
    until a synset with min_depth() <= depth_threshold is reached, then return
    the first lemma name.
    """
    try:
        # Convert wnid string (e.g., "n01440764") to integer offset (removing the 'n')
        synset = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
    except Exception as e:
        print("Error processing wnid:", wnid, e)
        return None
    while synset.hypernyms():
        hypernym = synset.hypernyms()[0]  # choose the first hypernym
        if hypernym.min_depth() <= depth_threshold:
            return hypernym.lemmas()[0].name()
        synset = hypernym
    return synset.lemmas()[0].name()


def get_imagenet_mapping(
    root="/datashare/ImageNet/ILSVRC2012", split="train", depth_threshold=5
):
    """
    Scans the ImageNet directory for the given split (e.g., 'train')
    and returns a mapping (dictionary) from fine label index (0 to 999)
    to a coarse label. This mapping is derived by processing the WordNet ID
    (folder name) and traversing its hypernym chain.

    Parameters:
        root (str): Root directory of the ImageNet dataset.
        split (str): Folder name for the split ('train' or 'val').
        depth_threshold (int): The minimum depth threshold for obtaining
                               a coarse label.

    Returns:
        dict: Mapping fine label index -> coarse label name.
    """
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise Exception(f"Directory {split_dir} does not exist.")

    # Get a sorted list of subdirectories (these are the WordNet IDs)
    wnids = sorted(
        [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    )

    mapping = {
        i: get_coarse_class(wnid, depth_threshold) for i, wnid in enumerate(wnids)
    }
    return mapping


##############################################################
# Helper function: Compute CIFAR-100 fine-to-coarse mapping
##############################################################
def get_cifar_mapping():
    """
    Returns a hard-coded mapping from fine labels (0-99) to coarse labels (0-19)
    for CIFAR-100.

    Since torchvision's CIFAR100 (version 0.9) does not provide coarse labels,
    we use the standard mapping.
    """
    fine_to_coarse = {
        0: 4,
        1: 1,
        2: 14,
        3: 8,
        4: 0,
        5: 6,
        6: 7,
        7: 7,
        8: 18,
        9: 3,
        10: 3,
        11: 14,
        12: 9,
        13: 18,
        14: 7,
        15: 11,
        16: 3,
        17: 9,
        18: 7,
        19: 11,
        20: 6,
        21: 11,
        22: 5,
        23: 10,
        24: 7,
        25: 6,
        26: 13,
        27: 15,
        28: 3,
        29: 15,
        30: 0,
        31: 11,
        32: 1,
        33: 10,
        34: 12,
        35: 14,
        36: 16,
        37: 9,
        38: 11,
        39: 5,
        40: 5,
        41: 19,
        42: 8,
        43: 8,
        44: 15,
        45: 13,
        46: 14,
        47: 13,
        48: 16,
        49: 13,
        50: 15,
        51: 13,
        52: 16,
        53: 19,
        54: 2,
        55: 4,
        56: 2,
        57: 0,
        58: 1,
        59: 4,
        60: 6,
        61: 19,
        62: 0,
        63: 2,
        64: 6,
        65: 19,
        66: 5,
        67: 7,
        68: 14,
        69: 15,
        70: 3,
        71: 8,
        72: 8,
        73: 15,
        74: 14,
        75: 2,
        76: 10,
        77: 11,
        78: 3,
        79: 13,
        80: 12,
        81: 16,
        82: 12,
        83: 17,
        84: 3,
        85: 17,
        86: 10,
        87: 16,
        88: 17,
        89: 4,
        90: 7,
        91: 5,
        92: 19,
        93: 2,
        94: 12,
        95: 12,
        96: 16,
        97: 16,
        98: 15,
        99: 13,
    }
    return fine_to_coarse


##############################################################
# Helper function: Invert mapping for continuous variant
##############################################################
def invert_mapping(fine_to_coarse):
    """
    Inverts a mapping from fine class to coarse label into a mapping from coarse label to a list of fine classes.
    """
    coarse_to_fine = {}
    for fine, coarse in fine_to_coarse.items():
        if coarse not in coarse_to_fine:
            coarse_to_fine[coarse] = []
        coarse_to_fine[coarse].append(fine)
    return coarse_to_fine
