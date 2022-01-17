from typing import Union, Optional, Dict, Tuple
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import shutil
import open3d as o3d
import numpy as np
from tqdm import tqdm

from .utils import (
    VoxelGrid,
    read_pointcloud_o3d, read_pointcloud_np,
    write_pointcloud_o3d, write_pointcloud_np,
    read_mesh,
)
from ._pcdmeshing import meshing_from_paths, meshing_from_paths_with_vis


def reconstruction_fn(args, max_edge_length: float = 1., max_visibility: int = 5):
    pcd_path, mesh_path, vis_paths = args
    if vis_paths is None:
        meshing_from_paths(
            str(pcd_path), str(mesh_path),
            max_edge_length=max_edge_length)
    else:
        endpoints, observations = vis_paths
        meshing_from_paths_with_vis(
            str(pcd_path), str(mesh_path), str(endpoints), str(observations),
            max_edge_length=max_edge_length, max_visibility=max_visibility)
    return True


def export_visibility_info(pcd_all_path: Path,
                           pcd_obs_path: Path,
                           tmp_vis_dir: Path,
                           grid: VoxelGrid,
                           margin: float,
                           points: np.ndarray,
                           vid2pidxs: Dict[int, np.ndarray]
                           ) -> Dict[int, Tuple[Path]]:
    tmp_paths = {}
    for vid in tqdm(vid2pidxs):
        key = '_'.join(map(str, grid.voxel_id_to_index(vid)))
        endpoints_path = tmp_vis_dir / (key+"_all.ply")
        obs_path = tmp_vis_dir / (key+"_obs.ply")
        tmp_paths[vid] = (endpoints_path, obs_path)
    if tmp_vis_dir.exists():
        return tmp_paths

    points_all = read_pointcloud_np(pcd_all_path)
    points_obs_all = read_pointcloud_np(pcd_obs_path)

    vid2pidxs_overlap_all = grid.voxelize_points_with_overlap(
        grid.voxelize_points(points_all), points_all, margin)
    dtype = o3d.core.Dtype.Float32

    tmp_vis_dir.mkdir(exist_ok=True)
    for vid in tqdm(vid2pidxs):
        indices_crop = vid2pidxs_overlap_all[vid]
        points_crop = points_all[indices_crop]
        points_block = points[vid2pidxs[vid]]

        nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points_crop, dtype=dtype))
        nns.hybrid_index()
        indices_knn, _ = nns.hybrid_search(o3d.core.Tensor(points_block, dtype=dtype), 0.01**2, 1)
        indices_knn = np.unique(indices_knn[:, 0].numpy())
        if indices_knn[0] == -1:
            indices_knn = indices_knn[1:]
        indices_block = indices_crop[indices_knn]

        endpoints_path, obs_path = tmp_paths[vid]
        write_pointcloud_np(endpoints_path, points_all[indices_block])
        write_pointcloud_np(obs_path, points_obs_all[indices_block])

    return tmp_paths



def run_block_meshing(pcd: Union[Path, o3d.geometry.PointCloud],
                      voxel_size: float = 20,
                      margin_seam: float = 0.2,
                      margin_discard: float = 0.2,
                      num_parallel: int = 10,
                      tmp_dir: Optional[Path] = None,
                      use_visibility: bool = False,
                      pcd_all_path: Optional[Path] = None,
                      pcd_obs_path: Optional[Path] = None,
                      opts: Dict = dict(max_edge_length=1., max_visibility=10),
                      cleanup: bool = True,
                      ) -> o3d.geometry.TriangleMesh:
    if tmp_dir is None:
        tmp_dir = Path("/tmp/block_meshing/")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    tmp_pcd_dir = tmp_dir / "blocks_pcd"
    tmp_mesh_dir = tmp_dir / "blocks_mesh"
    tmp_vis_dir = tmp_dir / "blocks_vis"

    margin_overlap = margin_seam + margin_discard
    if use_visibility:
        if pcd_all_path is None or not pcd_all_path.exists():
            raise ValueError(f"Incorrect endpoints file: {pcd_all_path}.")
        if pcd_obs_path is None or not pcd_obs_path.exists():
            raise ValueError(f"Incorrect observation file: {pcd_obs_path}.")

    if isinstance(pcd, (Path, str)):
        pcd = read_pointcloud_o3d(pcd)
    points = np.asarray(pcd.points, dtype=np.float32)
    grid = VoxelGrid(points.min(0), points.max(0), voxel_size=voxel_size)
    vid2pidxs_overlap = grid.voxelize_points_with_overlap(
        grid.voxelize_points(points), points, margin_overlap)

    tmp_filenames = {vid: '_'.join(map(str, grid.voxel_id_to_index(vid))) + ".ply"
                     for vid in vid2pidxs_overlap}
    tmp_pcd_dir.mkdir(parents=True, exist_ok=True)
    for vid, pidxs in tqdm(vid2pidxs_overlap.items()):
        path = tmp_pcd_dir / tmp_filenames[vid]
        if path.exists():
            continue
        pcd_block = pcd.select_by_index(pidxs)
        write_pointcloud_o3d(path, pcd_block)
        del pcd_block
    del pcd

    if use_visibility:
        tmp_vis_paths = export_visibility_info(
            pcd_all_path, pcd_obs_path, tmp_vis_dir, grid,
            margin_overlap, points, vid2pidxs_overlap)
    args = [(tmp_pcd_dir/tmp_filenames[vid], tmp_mesh_dir/tmp_filenames[vid])
            + (tmp_vis_paths[vid] if use_visibility else None,)
            for vid in vid2pidxs_overlap]
    tmp_mesh_dir.mkdir(exist_ok=True)
    with Pool(processes=num_parallel) as pool:
        ret = list(tqdm(pool.imap(partial(reconstruction_fn, **opts), args), total=len(args)))
    assert all(ret), ret

    bbox_size = grid.voxel_size // 2 + margin_seam
    mesh_total = None
    for vid in tqdm(vid2pidxs_overlap):
        mesh_block = read_mesh(tmp_mesh_dir / tmp_filenames[vid])
        center = grid.voxel_center_from_index(grid.voxel_id_to_index(vid))
        bbox = o3d.geometry.AxisAlignedBoundingBox(center-bbox_size, center+bbox_size)
        mesh_block = mesh_block.crop(bbox)
        if mesh_total is None:
            mesh_total = mesh_block
        else:
            mesh_total += mesh_block
        del mesh_block

    mesh_total = mesh_total.merge_close_vertices(1e-3).remove_degenerate_triangles()

    if cleanup:
        shutil.rmtree(str(tmp_dir))

    return mesh_total
