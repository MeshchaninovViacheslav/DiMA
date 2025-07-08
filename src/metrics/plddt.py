import os
import torch
import biotite.structure.io as bsio
from typing import List, Dict
from tqdm import tqdm

from cheap.esmfold import esmfold_v1


class ESMMetric:
    def __init__(self, device: str = "cpu"):
        self.model = esmfold_v1()
        self.model = self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, protein: str, index: int, pdb_path: str) -> float:
        if not protein:
            return 0
        output = self.model.infer_pdb(protein)

        os.makedirs(pdb_path, exist_ok=True)
        file_path = os.path.join(pdb_path, f"{index:05d}.pdb")

        with open(file_path, "w") as f:
            f.write(output)
        struct = bsio.load_structure(file_path, extra_fields=["b_factor"])
        return struct.b_factor.mean()


def calculate_plddt(predictions: List[str], index_list: List[int], device="cuda", pdb_path="") -> Dict[str, float]:
    metric_fn = ESMMetric(device)

    os.makedirs(pdb_path, exist_ok=True)

    result = dict()
    for i, protein in tqdm(enumerate(predictions)):
        ind = index_list[i]
        result[protein] = metric_fn(protein=protein, index=ind, pdb_path=pdb_path)
    return result