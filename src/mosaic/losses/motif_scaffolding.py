import jax.nn as nn
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Array

from mosaic.common import LossTerm, restype_three_to_one
from mosaic.structure_prediction import AbstractStructureOutput
from mosaic.util import kabsch, gram_schmidt                               

class DistogramCCE(LossTerm):
    gt_distogram: Float[Array, "N N"]
    mask: Float[Array, "N"]

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        # Only keep scaffold positions
        pred_logits = output.distogram_logits[self.mask][:, self.mask]
        gt_distogram = self.gt_distogram[self.mask][:, self.mask] 
        bins = output.distogram_bins
        # Turn ground-truth to one-hot
        gt_indices = jnp.digitize(gt_distogram, bins)
        gt_indices = jnp.clip(gt_indices, 0, pred_logits.shape[-1] - 1)
        gt_one_hot = nn.one_hot(gt_indices, num_classes=pred_logits.shape[-1])
        
        # Compute CCE
        loss = -jnp.sum(gt_one_hot * nn.log_softmax(pred_logits, axis=-1), axis=-1)
        dgramm_cce = jnp.mean(loss)
        return dgramm_cce, {"dgramm_cce": dgramm_cce}

class FAPE(LossTerm):
    gt_t: Float[Array, "N 3"]
    gt_R: Float[Array, "N 3 3"]
    mask: Float[Array, "N"]

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        def robust_norm(x, eps=1e-8):
            return jnp.sqrt(jnp.square(x).sum(axis=-1) + eps)

        def get_ij(R, t):
            return jnp.einsum("rji, rsj -> rsi", R, t[None, :] - t[:, None])

        # Only keep scaffold positions
        pred_bb = output.backbone_coordinates[self.mask]      
        gt_R  = self.gt_R[self.mask]      
        gt_t  = self.gt_t[self.mask]   

        # Compute rotation on prediction backbones
        pred_R = gram_schmidt(
            v1=pred_bb[:, 2, :] - pred_bb[:, 1, :],   # CA -> C
            v2=pred_bb[:, 0, :] - pred_bb[:, 1, :],   # CA -> N
        )                                             
        pred_t = pred_bb[:, 1, :]                   
        pred_ij = get_ij(pred_R, pred_t)    
        gt_ij = get_ij(gt_R, gt_t)              

        # Compute FAPE
        fape = robust_norm(pred_ij - gt_ij)    
        fape = jnp.clip(fape, 0.0, 10.0) / 10.0
        fape = fape.mean()

        return fape, {"fape": fape}                                

class RMSD(LossTerm):
    gt_coords: Float[Array, "N 4 3"]
    mask:      Float[Array, "N"]

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: AbstractStructureOutput,
        key,
    ):
        # Only keep scaffold positions
        pred_bb = output.backbone_coordinates[self.mask] 
        gt_bb = self.gt_coords[self.mask]               
        # Align pred to gt
        pred = pred_bb.reshape(-1, 3)                      
        gt = gt_bb.reshape(-1, 3)                    
        R, t = kabsch(pred, gt)
        pred_aligned = pred @ R + t
        # Compute RMSD
        rmsd = jnp.sqrt(jnp.mean(jnp.sum((pred_aligned - gt) ** 2, axis=-1)))
        return rmsd, {"scaffold_rmsd": rmsd}
    
import biotite.structure as struct
from biotite.structure.io import pdb, pdbx
    
class Scaffold():
    def __init__(self,
                 path_to_structure: str,
                 keep_intervals: list[tuple[int, int]],
                 order: list[int],
                 loops: list[int],
                 ):
        assert len(loops) == len(order) + 1, \
            f"loops must have length len(order)+1={len(order)+1}, got {len(loops)}"

        assert len(order) == len(keep_intervals) and max(order) == len(keep_intervals) - 1, \
            f"order must include an idx for each element of keep_intervals " \
            f"starting from 0 to len(keep_intervals)-1={len(keep_intervals)-1}"

        # Load and filter to amino acids
        if path_to_structure.endswith(".pdb"):
            pdb_file = pdb.PDBFile.read(path_to_structure)
            structure = pdb.get_structure(pdb_file, model=1)
        elif path_to_structure.endswith(".cif"):
            cif_file = pdbx.PDBxFile.read(path_to_structure)
            structure = pdbx.get_structure(cif_file, model=1)
        else:
            raise ValueError("File must be .pdb or .cif")

        structure = structure[struct.filter_amino_acids(structure)]
        bb_atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3}

        def extract_fragment(start, end):
            frag = structure[(structure.res_id >= start) & (structure.res_id <= end)]

            bb_mask  = np.isin(frag.atom_name, ["N", "CA", "C", "O"])
            bb_atoms = frag[bb_mask]
            sort_key = np.array([bb_atom_order.get(n, 99) for n in bb_atoms.atom_name])
            sort_idx = np.lexsort((sort_key, bb_atoms.res_id))
            bb_coords = bb_atoms.coord[sort_idx].reshape(-1, 4, 3)

            ca_mask  = frag.atom_name == "CA"
            ca_atoms = frag[ca_mask]
            ca_coords = ca_atoms.coord
            sequence  = "".join(
                restype_three_to_one.get(name, "X") for name in ca_atoms.res_name
            )

            return bb_coords, ca_coords, sequence

        def make_loop(n):
            bb  = np.zeros((n, 4, 3), dtype=np.float32)
            ca  = np.zeros((n, 3),    dtype=np.float32)
            seq = "X" * n
            return bb, ca, seq

        # Assemble dummy sequence
        all_bb, all_ca, all_seq, all_mask = [], [], [], []

        for i, frag_idx in enumerate(order):
            if loops[i] > 0:
                bb, ca, seq = make_loop(loops[i])
                all_bb.append(bb);   all_ca.append(ca)
                all_seq.append(seq); all_mask.append(np.zeros(loops[i], dtype=bool))

            start, end = keep_intervals[frag_idx]
            bb, ca, seq = extract_fragment(start, end)
            L = ca.shape[0]
            all_bb.append(bb);   all_ca.append(ca)
            all_seq.append(seq); all_mask.append(np.ones(L, dtype=bool))

        if loops[-1] > 0:
            bb, ca, seq = make_loop(loops[-1])
            all_bb.append(bb);   all_ca.append(ca)
            all_seq.append(seq); all_mask.append(np.zeros(loops[-1], dtype=bool))

        # Store coords and others
        bb_coords = np.concatenate(all_bb,   axis=0)  # [L_total, 4, 3]
        ca_coords = np.concatenate(all_ca,   axis=0)  # [L_total, 3]
        valid     = np.concatenate(all_mask, axis=0)  # [L_total]

        self.full_sequence = "".join(all_seq)
        self.mask          = jnp.array(valid)            # [L_total] bool
        self._bb_coords    = jnp.array(bb_coords)        # [L_total, 4, 3]
        self._ca_coords    = jnp.array(ca_coords)        # [L_total, 3]

    def __len__(self) -> int:
        return len(self.full_sequence)

    def distogram(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        diff = self._ca_coords[:, None, :] - self._ca_coords[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1)
        pair_mask = self.mask[:, None] & self.mask[None, :]
        return jnp.where(pair_mask, dist, 0.0)

    def backbone_coordinates(self) -> jnp.ndarray:
        return self._bb_coords

    def backbone_frames(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        n_coords  = self._bb_coords[:, 0, :]
        ca_coords = self._bb_coords[:, 1, :]
        c_coords  = self._bb_coords[:, 2, :]

        R = gram_schmidt(v1=c_coords - ca_coords, v2=n_coords - ca_coords)
        identity = jnp.broadcast_to(jnp.eye(3), R.shape)
        R = jnp.where(self.mask[:, None, None], R, identity)
        t = jnp.where(self.mask[:, None], ca_coords, 0.0)
        return t, R