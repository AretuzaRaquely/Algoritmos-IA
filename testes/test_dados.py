import numpy as np

from src.dados.coco17 import COCO17_ARTICULACOES, NUMERO_ARTICULACOES


def test_coordenadas_coco17():
    assert len(COCO17_ARTICULACOES) == NUMERO_ARTICULACOES == 17


def test_pose_real_shape():
    pose = np.zeros((17, 2), dtype=np.float32)
    assert pose.shape == (17, 2)
