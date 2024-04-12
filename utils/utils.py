import logging

import numpy as np


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def calculate_winner(bbox_predictions):
    weights = {
        0: -1,
        1: 1,
        2: 0,
        None: 0,
    }

    left_bbox = bbox_predictions["bbox_left"]
    right_bbox = bbox_predictions["bbox_right"]

    def calculate_score(competitor):
        score = [weights[num] for num in competitor]
        return sum(score)

    score_left = calculate_score(left_bbox)
    score_right = calculate_score(right_bbox)

    move_decision = ""

    with open("score.txt", "w") as f:
        if score_left > score_right:
            move_decision = "E"
        else:
            move_decision = "D"
        f.write(f"{move_decision}")

    return "Esquerda" if move_decision == "E" else "Direita"
