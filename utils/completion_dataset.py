"""
MeshCompletionDataset – generates (partial_mesh + mask) -> complete_mesh training pairs.

On-the-fly masking: every __getitem__ call applies a random mask to the complete mesh,
so the model sees different partial patterns each epoch (free data augmentation).

The dataset also saves a fixed set of test samples (both partial + GT complete) to disk
so you can visually inspect them and compute errors later.
"""

import os
import os.path as osp
import random
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from trimesh import Trimesh, load_mesh
from concurrent.futures import ProcessPoolExecutor
from glob import glob

from utils.funcs import save_ply_explicit, get_edge_index
from utils.masking import generate_random_mask, _build_adjacency


def remove_masked_region(verts, faces, mask):
    """Drop masked (missing) vertices and every face that touches one, then
    re-index the survivors. Returns (new_verts, new_faces) forming a clean
    partial surface — the remaining region stays intact with a real hole where
    the missing part was (NO spikes/fans to the origin).

    mask: (N,) bool, True = MISSING.
    """
    verts = np.asarray(verts)
    faces = np.asarray(faces)
    keep = ~np.asarray(mask, dtype=bool)                 # (N,) True = kept
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    face_keep = keep[faces].all(axis=1)                  # faces with all 3 kept
    new_faces = remap[faces[face_keep]]
    new_verts = verts[keep]
    return new_verts, new_faces


class Normalize3D(object):
    """Normalize xyz coordinates (first 3 channels of x and all of y)."""

    def __init__(self, mean, std=None, mean_subtraction_only=True):
        self.mean = mean
        self.std = None if mean_subtraction_only else std
        self.mean_subtraction_only = mean_subtraction_only

    def __call__(self, data):
        mean = torch.as_tensor(self.mean, dtype=data.y.dtype, device=data.y.device)

        # Normalize target y (complete mesh, 3 channels)
        data.y = data.y - mean
        if not self.mean_subtraction_only and self.std is not None:
            std = torch.as_tensor(self.std, dtype=data.y.dtype, device=data.y.device)
            data.y = data.y / std

        data.y.nan_to_num_(0.0)
        return data


class MeshCompletionDataset(InMemoryDataset):
    """
    Dataset for mesh completion training.

    Input  x : (N, 4) = [x, y, z, mask]  where mask=1 for known, 0 for missing
    Target y : (N, 3) = complete vertex positions (normalized)

    During process(), a fixed test split is saved along with GT and partial .ply files.
    At training time, masking is applied on-the-fly via transform.
    """

    def __init__(self, config, dtype='train', mean_subtraction_only=True):
        assert dtype in ['train', 'eval'], "dtype must be 'train' or 'eval'"
        self.config = config
        self.root = config['dataset_dir']
        self.dtype = dtype
        self.mean_subtraction_only = mean_subtraction_only

        # Masking params from config (with conservative defaults)
        self.mask_ratio_min = config.get('mask_ratio_min', 0.05)
        self.mask_ratio_max = config.get('mask_ratio_max', 0.15)
        # Strategy selection (bodies use contiguous-only; see config).
        self.mask_strategies = config.get(
            'mask_strategies', ('bfs_patch', 'plane_cut', 'random_scatter'))
        self.mask_strategy_weights = config.get(
            'mask_strategy_weights', (0.5, 0.3, 0.2))

        super(MeshCompletionDataset, self).__init__(self.root)

        idx = 0 if dtype == 'train' else 1
        self.data, self.slices = torch.load(self.processed_paths[idx], weights_only=False)
        norm_dict = torch.load(self.processed_paths[2], weights_only=False)
        self.mean = norm_dict['mean']
        self.std = norm_dict['std']
        self.edge_index = torch.load(self.processed_paths[3], weights_only=False)

        # Store faces for masking (loaded once)
        template_mesh = load_mesh(config['template_file'])
        self.faces_np = np.array(template_mesh.faces)

        # Precompute vertex adjacency ONCE (topology is fixed). On-the-fly masking
        # reuses this instead of rebuilding it from faces on every __getitem__,
        # which was a major CPU bottleneck (and made the whole machine sluggish).
        n_verts = int(self.faces_np.max()) + 1
        self._adj = _build_adjacency(self.faces_np, n_verts)

        # Geodesic depth-weighting (optional). The solver caches the edge graph
        # built from the FIXED template, so each __getitem__ only pays for one
        # multi-source shortest-path solve (~2 ms on COMA) instead of rebuilding
        # the graph. Same reasoning as _adj above.
        self.depth_weight = bool(config.get('depth_weight', False))
        self._bfield = None
        if self.depth_weight:
            from utils.boundary_field import BoundaryFieldSolver
            self.depth_weight_max = config.get('depth_weight_max',
                                               config.get('lambda_missing', 3.0))
            self.depth_weight_normalize = bool(config.get('depth_weight_normalize', False))
            self._bfield = BoundaryFieldSolver(
                self.faces_np, np.array(template_mesh.vertices),
                backend=config.get('depth_weight_backend', 'dijkstra'))

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed_completion')

    @property
    def processed_file_names(self):
        return ['training.pt', 'eval.pt', 'norm.pt', 'edge_index.pt']

    def process(self):
        meshes = []
        train_data, eval_data = [], []
        train_vertices = []
        train_dir = self.config.get('train_dir') or osp.join(self.root, "train")

        train_list = self.config.get('train_list')
        if train_list and osp.exists(train_list):
            # explicit file list from prepare_data.py (the split's train + val meshes)
            with open(train_list) as fh:
                files = sorted(line.strip() for line in fh if line.strip())
            print(f"Found {len(files)} mesh files in {train_list}")
        else:
            # Find all ply files
            files = sorted(glob(osp.join(train_dir, '**', '*.ply'), recursive=True))
            print(f"Found {len(files)} mesh files in {train_dir}")

        # Load meshes
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(load_mesh, f) for f in files]
            for future in tqdm(futures, desc="Loading meshes"):
                meshes.append(future.result())

        edge_index = get_edge_index(meshes[0].vertices, meshes[0].faces)
        faces_np = np.array(meshes[0].faces)

        # Shuffle with a fixed seed so the split is reproducible
        indices = list(range(len(meshes)))
        random.shuffle(indices)
        count_train = int(0.8 * len(meshes))

        # ---- Save test meshes (GT complete + partial examples) ----
        test_save_dir = osp.join(self.root, 'completion_test_samples')
        os.makedirs(test_save_dir, exist_ok=True)
        rng_test = np.random.RandomState(42)  # fixed seed for reproducible test masks

        num_test_saved = 0
        max_test_to_save = min(20, len(meshes) - count_train)  # save up to 20 test pairs

        for idx_in_shuffled, orig_idx in enumerate(indices):
            mesh_verts = torch.Tensor(meshes[orig_idx].vertices)

            # x=y=complete vertices for now; masking happens on-the-fly during training
            data = Data(x=mesh_verts.clone(), y=mesh_verts.clone(), edge_index=edge_index)

            if idx_in_shuffled < count_train:
                train_data.append(data)
                train_vertices.append(mesh_verts)
            else:
                eval_data.append(data)

                # Save test pairs (GT + partial) for visual inspection
                if num_test_saved < max_test_to_save:
                    verts_np = meshes[orig_idx].vertices
                    mask, strategy, ratio = generate_random_mask(
                        verts_np, faces_np,
                        mask_ratio_range=(self.config.get('mask_ratio_min', 0.05),
                                          self.config.get('mask_ratio_max', 0.15)),
                        strategies=self.config.get('mask_strategies',
                                                   ('bfs_patch', 'plane_cut', 'random_scatter')),
                        strategy_weights=self.config.get('mask_strategy_weights',
                                                         (0.5, 0.3, 0.2)),
                        rng=rng_test
                    )

                    # Save GT complete mesh (process=False to keep exact vertex order)
                    gt_mesh = Trimesh(vertices=verts_np.copy(), faces=faces_np.copy(), process=False)
                    gt_path = osp.join(test_save_dir, f'test_{num_test_saved:03d}_gt_complete.ply')
                    save_ply_explicit(gt_mesh, gt_path)

                    # Save partial mesh: actually REMOVE the missing vertices and any
                    # face touching them (re-indexed), so the remaining surface stays
                    # intact with a clean hole -- no origin spikes/fans.
                    pv, pf = remove_masked_region(verts_np, faces_np, mask)
                    partial_mesh = Trimesh(vertices=pv, faces=pf, process=False)
                    partial_path = osp.join(test_save_dir, f'test_{num_test_saved:03d}_partial_{strategy}_r{ratio:.2f}.ply')
                    save_ply_explicit(partial_mesh, partial_path)

                    # Save mask as npy for error computation
                    mask_path = osp.join(test_save_dir, f'test_{num_test_saved:03d}_mask.npy')
                    np.save(mask_path, mask)

                    num_test_saved += 1

        print(f"Saved {num_test_saved} test sample pairs in {test_save_dir}")
        print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

        # Compute normalization stats from training data (xyz only)
        mean_train = torch.mean(torch.stack(train_vertices), dim=0)
        std_train = torch.std(torch.stack(train_vertices), dim=0)
        std_train[std_train == 0] = 1.0

        if self.mean_subtraction_only:
            norm_dict = {'mean': mean_train, 'std': None}
        else:
            norm_dict = {'mean': mean_train, 'std': std_train}

        # Save template mesh (mean shape)
        template_mesh = Trimesh(vertices=mean_train.numpy(), faces=faces_np)
        save_ply_explicit(template_mesh, self.config['template_file'])

        # Normalize targets (y = complete vertices)
        transform = Normalize3D(mean_train, std_train, mean_subtraction_only=self.mean_subtraction_only)
        train_data = [transform(d) for d in train_data]
        eval_data = [transform(d) for d in eval_data]

        # x is also stored as normalized complete vertices; masking applied on-the-fly
        # So normalize x the same way as y for storage
        for d in train_data + eval_data:
            mean = torch.as_tensor(mean_train, dtype=d.x.dtype)
            d.x = d.x - mean
            if not self.mean_subtraction_only:
                std = torch.as_tensor(std_train, dtype=d.x.dtype)
                d.x = d.x / std
            d.x.nan_to_num_(0.0)

        # Save
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(eval_data), self.processed_paths[1])
        torch.save(norm_dict, self.processed_paths[2])
        torch.save(edge_index, self.processed_paths[3])
        torch.save(norm_dict, osp.join(self.root, "norm_completion.pt"))

        print("Completion dataset processing done.")

    def set_mask_ratio_range(self, ratio_min, ratio_max):
        """Update mask ratio range dynamically (used by curriculum learning)."""
        self.mask_ratio_min = ratio_min
        self.mask_ratio_max = ratio_max

    def get(self, idx):
        """
        Override get() to apply random masking on-the-fly.
        Stored x is normalized complete vertices (N, 3).
        We create partial input (N, 4) = [masked_xyz, mask_channel].
        """
        data = super().get(idx)

        # data.x = (N, 3) normalized complete vertices
        # data.y = (N, 3) normalized complete vertices (target)
        verts_normalized = data.x.clone()  # (N, 3)
        N = verts_normalized.shape[0]

        # Generate random mask (on un-normalized vertices doesn't matter for mask selection,
        # we use the normalized positions — the spatial relationships are preserved)
        rng = np.random.RandomState()
        mask, _, _ = generate_random_mask(
            verts_normalized.numpy(), self.faces_np,
            mask_ratio_range=(self.mask_ratio_min, self.mask_ratio_max),
            strategies=self.mask_strategies,
            strategy_weights=self.mask_strategy_weights,
            rng=rng, adj=self._adj
        )

        # Create mask channel: 1 = known, 0 = missing
        mask_tensor = torch.ones(N, 1, dtype=verts_normalized.dtype)
        mask_tensor[mask] = 0.0

        # Zero out missing vertex coordinates
        partial_verts = verts_normalized.clone()
        partial_verts[mask] = 0.0

        # Concatenate: (N, 4) = [x, y, z, mask]
        data.x = torch.cat([partial_verts, mask_tensor], dim=1)

        # Store the boolean mask for loss weighting
        data.vertex_mask = torch.tensor(mask, dtype=torch.bool)  # True = missing

        # Optional per-vertex geodesic depth weight. Attached only when enabled;
        # when absent the loss falls back to the flat lambda_known/lambda_missing
        # split, so the existing arms are unaffected.
        if self._bfield is not None:
            w = self._bfield.depth_weight(
                mask,
                w_max=self.depth_weight_max,
                w_known=self.config.get('lambda_known', 1.0),
                normalize=self.depth_weight_normalize)
            data.loss_weight = torch.tensor(w, dtype=verts_normalized.dtype)

        return data
