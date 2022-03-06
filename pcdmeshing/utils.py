from typing import Dict
from pathlib import Path
import numpy as np
import open3d as o3d
from tqdm import tqdm
import plyfile


class VoxelGrid:
    def __init__(self, min_: np.ndarray, max_: np.ndarray, voxel_size: float):
        self.origin = min_ - voxel_size / 2
        self.voxel_size = voxel_size
        self.num_indices = self.xyz_to_voxel_index(max_) + 1

    def xyz_to_voxel_index(self, points: np.ndarray) -> np.ndarray:
        '''A voxel index is a 3-dimensional integer index.'''
        return np.floor((points - self.origin) / self.voxel_size).astype(np.int32)

    def voxel_index_to_id(self, vidxs: np.ndarray) -> np.ndarray:
        '''A voxel ID is a scalar integer index.'''
        xm, ym, _ = self.num_indices
        return vidxs[..., 0] + xm*vidxs[..., 1] + xm*ym*vidxs[..., 2]

    def voxel_id_to_index(self, vids: np.ndarray) -> np.ndarray:
        xm, ym, _ = self.num_indices
        k = vids // (xm*ym)
        vids = vids % (xm*ym)
        j = vids // (xm)
        i = vids % xm
        return np.stack([i, j, k], -1)

    def voxel_center_from_index(self, vidx: np.ndarray) -> np.ndarray:
        return (vidx + 0.5)*self.voxel_size + self.origin

    def voxelize_points(self, points: np.ndarray) -> Dict[int, np.ndarray]:
        '''Create a mapping from voxel ID to indices of 3D points in the voxel.
           Memory efficient by storing the indices as uint32 numpy arrays.'''
        pidx_to_vidx = self.xyz_to_voxel_index(points)
        valid = np.all(pidx_to_vidx >= 0, -1) & np.all(pidx_to_vidx < self.num_indices, -1)
        pidx_to_vid = self.voxel_index_to_id(pidx_to_vidx)
        vids_unique, vids_counts = np.unique(pidx_to_vid[valid], return_counts=True)

        vid2pidxs = dict()
        for i, vid in enumerate(tqdm(vids_unique)):
            vid2pidxs[vid] = np.zeros(vids_counts[i], dtype=np.uint32)
        del vids_unique
        del vids_counts

        vid2pidxs_offset = {vid: 0 for vid in vid2pidxs}
        for i, pidx in enumerate(tqdm(np.arange(len(pidx_to_vid), dtype=np.uint32))):
            if not valid[i]:
                continue
            vid = pidx_to_vid[i]
            vid2pidxs[vid][vid2pidxs_offset[vid]] = pidx
            vid2pidxs_offset[vid] += 1
        del vid2pidxs_offset

        return vid2pidxs

    def voxelize_points_with_overlap(self,
                                     vid2pidxs: Dict[int, np.ndarray],
                                     points: np.ndarray, margin: float
                                     ) -> Dict[int, np.ndarray]:
        '''Create a mapping from voxel ID to indices of 3D points in a bounding box centered
           at the voxel. Fast search based on neighbording voxels.'''
        assert margin < self.voxel_size
        query_size = self.voxel_size / 2 + margin
        vid2pidxs_overlap = {}

        for vid in tqdm(vid2pidxs):
            vidx = self.voxel_id_to_index(vid)
            pidxs = vid2pidxs[vid]
            pidxs_search = []
            for i in range(3):
                for o in [-1, 0, 1]:
                    vidx_q = vidx.copy()
                    vidx_q[i] += o
                    if np.all(vidx_q == vidx):
                        continue
                    vid_q = self.voxel_index_to_id(vidx_q)
                    if vid_q not in vid2pidxs:
                        continue
                    pidxs_search.append(vid2pidxs[vid_q])
            if len(pidxs_search) != 0:
                pidxs_search = np.concatenate(pidxs_search)
                points_search = points[pidxs_search]
                center = self.voxel_center_from_index(vidx)
                mask = (np.all((center - query_size) <= points_search, -1)
                        & np.all(points_search <= (center + query_size), -1))
                pidxs_overlap = pidxs_search[mask]
                pidxs = np.concatenate([pidxs_overlap, pidxs])
            vid2pidxs_overlap[vid] = pidxs

        return vid2pidxs_overlap


def read_pointcloud_o3d(path: Path) -> o3d.geometry.PointCloud:
    '''Reads also colors and normals but stores the points at float64.'''
    return o3d.io.read_point_cloud(str(path))


def read_pointcloud_np(path: Path) -> np.ndarray:
    '''Only reads the point coordinates but preserves the float32 dtype.'''
    pcd = o3d.t.io.read_point_cloud(str(path))
    return np.asarray(pcd.point["points"].numpy())


def write_pointcloud_o3d(path: Path, pcd: o3d.geometry.PointCloud,
                         write_normals: bool = True, xyz_dtype: str = 'float32'):
    '''Currently o3d.t.io.write_point_cloud writes non-standard types but #4553 should fixe it.'''
    # pcd_t = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.asarray(pcd.points)))
    # if pcd.has_normals():
        # pcd_t.point["normals"] = o3d.core.Tensor(np.asarray(pcd.normals))
    # if pcd.has_colors():
        # pcd_t.point["colors"] = o3d.core.Tensor(np.asarray(pcd.colors), dtype=o3d.core.Dtype.UInt8)
    # o3d.t.io.write_point_cloud(str(path), pcd_t)
    write_normals = write_normals and pcd.has_normals()
    dtypes = [('x', xyz_dtype), ('y', xyz_dtype), ('z', xyz_dtype)]
    if write_normals:
        dtypes.extend([('nx', xyz_dtype), ('ny', xyz_dtype), ('nz', xyz_dtype)])
    if pcd.has_colors():
        dtypes.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    data = np.empty(len(pcd.points), dtype=dtypes)
    data['x'], data['y'], data['z'] = np.asarray(pcd.points).T
    if write_normals:
        data['nx'], data['ny'], data['nz'] = np.asarray(pcd.normals).T
    if pcd.has_colors():
        colors = (np.asarray(pcd.colors)*255).astype(np.uint8)
        data['red'], data['green'], data['blue'] = colors.T
    with open(str(path), mode='wb') as f:
        plyfile.PlyData([plyfile.PlyElement.describe(data, 'vertex')]).write(f)


def write_pointcloud_np(path: Path, points: np.ndarray):
    '''Only writes the point coordinates but preserves the float32 dtype.'''
    o3d.t.io.write_point_cloud(
        str(path),
        o3d.t.geometry.PointCloud(o3d.core.Tensor(points)))


def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    return o3d.io.read_triangle_mesh(str(path))


def write_mesh(path: Path, mesh: o3d.geometry.TriangleMesh):
    o3d.io.write_triangle_mesh(
        str(path), mesh, compressed=True, print_progress=True)
