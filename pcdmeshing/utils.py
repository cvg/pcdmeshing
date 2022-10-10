from typing import Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from tqdm import tqdm
import plyfile


@dataclass
class PointCloud:
    points: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.points.ndim == 2 and self.points.shape[-1] == 3
        if self.colors is not None:
            assert self.colors.ndim == 2 and self.colors.shape[-1] == 3
            assert len(self.points) == len(self.colors)
        if self.normals is not None:
            assert self.normals.ndim == 2 and self.normals.shape[-1] == 3
            assert len(self.points) == len(self.normals)

    @classmethod
    def from_o3d(cls, pcd: o3d.geometry.PointCloud):
        points = np.asarray(pcd.points).astype(np.float32, copy=False)
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors)*255).astype(np.uint8)
        else:
            colors = None
        if pcd.has_normals():
            normals = np.asarray(pcd.normals).astype(np.float32, copy=False)
        else:
            normals = None
        return cls(points, colors, normals)

    @classmethod
    def read(cls, path: Path, read_normals: bool = True, read_colors: bool = True):
        data = plyfile.PlyData.read(path)['vertex']
        points = np.stack([np.asarray(data[i]) for i in ('x', 'y', 'z')], -1)
        if read_normals and all(i in data for i in ('nx', 'ny', 'nz')):
            normals = np.stack([np.asarray(data[i]) for i in ('nx', 'ny', 'nz')], -1)
        else:
            normals = None
        if read_colors and all(i in data for i in ('red', 'green', 'blue')):
            colors = np.stack([np.asarray(data[i]) for i in ('red', 'green', 'blue')], -1)
        else:
            colors = None
        return cls(points, colors, normals)

    def write(self, path: Path, write_colors: bool = True):
        write_colors = write_colors and self.colors is not None
        write_normals = self.normals is not None
        dtypes = [(i, self.points.dtype) for i in ('x', 'y', 'z')]
        if write_colors:
            dtypes.extend([(i, self.colors.dtype) for i in ('red', 'green', 'blue')])
        if write_normals:
            dtypes.extend([(i, self.normals.dtype) for i in ('nx', 'ny', 'nz')])
        data = np.empty(len(self.points), dtype=dtypes)
        data['x'], data['y'], data['z'] = self.points.T
        if write_colors:
            data['red'], data['green'], data['blue'] = self.colors.T
        if write_normals:
            data['nx'], data['ny'], data['nz'] = self.normals.T
        with open(str(path), mode='wb') as f:
            plyfile.PlyData([plyfile.PlyElement.describe(data, 'vertex')]).write(f)

    def select_by_index(self, indices):
        return self.__class__(
            self.points[indices],
            None if self.colors is None else self.colors[indices],
            None if self.normals is None else self.normals[indices],
        )


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    colors: Optional[np.ndarray] = None

    @classmethod
    def read(cls, path: Path):
        data = plyfile.PlyData.read(path)
        vertices = data['vertex']
        points = np.stack([np.asarray(vertices[i]) for i in ('x', 'y', 'z')], -1)
        if all(i in vertices for i in ('red', 'green', 'blue')):
            colors = np.stack([np.asarray(vertices[i]) for i in ('red', 'green', 'blue')], -1)
        else:
            colors = None
        faces = np.vstack(data['face']['vertex_indices'])
        return cls(points, faces, colors)

    def write(self, path: Path):
        write_colors = self.colors is not None
        dtypes = [(i, self.vertices.dtype) for i in ('x', 'y', 'z')]
        if write_colors:
            dtypes.extend([(i, self.colors.dtype) for i in ('red', 'green', 'blue')])
        vertices = np.empty(len(self.vertices), dtype=dtypes)
        vertices['x'], vertices['y'], vertices['z'] = self.vertices.T
        if write_colors:
            vertices['red'], vertices['green'], vertices['blue'] = self.colors.T
        faces = np.empty(len(self.faces), dtype=[('vertex_indices', self.faces.dtype, (3,))])
        faces['vertex_indices'] = self.faces
        with open(str(path), mode='wb') as f:
            plyfile.PlyData([
                plyfile.PlyElement.describe(vertices, 'vertex'),
                plyfile.PlyElement.describe(faces, 'face'),
            ]).write(f)

    def to_o3d(self) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.vertices.astype(np.float64, copy=False)),
            o3d.utility.Vector3iVector(self.faces))
        mesh.vertex_colors = o3d.utility.Vector3dVector(self.colors/255.)
        return mesh

    def crop(self, min_, max_):
        keep = np.all(self.vertices >= min_, 1) & np.all(self.vertices <= max_, 1)
        new_indices = np.full(len(self.vertices), -1, self.faces.dtype)
        new_indices[keep] = np.arange(np.count_nonzero(keep), dtype=new_indices.dtype)
        keep_faces = keep[self.faces].all(1)
        return self.__class__(
            self.vertices[keep],
            new_indices[self.faces[keep_faces]],
            None if self.colors is None else self.colors[keep],
        )

    def __add__(self, other):
        vertices = np.concatenate([self.vertices, other.vertices], 0)
        faces = np.concatenate([self.faces, other.faces+len(self.vertices)], 0)
        if self.colors is None or other.colors is None:
            colors = None
        else:
            colors = np.concatenate([self.colors, other.colors], 0)
        return self.__class__(vertices, faces, colors)


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
