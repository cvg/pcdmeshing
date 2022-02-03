# Fast meshing of clean point clouds in Python

`pcdmeshing` is a Python package to reconstruct meshes from point clouds using [CGAL](https://github.com/CGAL/cgal).
- Based on the [Advancing Front surface reconstruction algorithm](https://doc.cgal.org/4.11.3/Advancing_front_surface_reconstruction/index.html#Chapter_Advancing_Front_Surface_Reconstruction) by [Cohen-Steiner & Da, The Visual Computer, 2004].
- Optionally uses point visibility, e.g. from Lidar, to constrain or cleanup the reconstruction.
- Scales to large scenes by performing block-wise parallel reconstruction.
- Compared to Poisson surface reconstruction, AF does not assume watertightness, does not resample the point cloud, and does not work at a fixed resolution. It is thus more accurate and produces fewer artifacts if the input point cloud is noise-free.


## Installation

1. Clone the repository and its submodules by running:

```sh
git clone --recursive git@github.com:cvg/pcdmeshing.git
cd pcdmeshing
```

2. Install Boost>=1.71. The builds of the Ubuntu package manager `libboost-all-dev` is usually too old, so we build from the header-only sources:

```sh
wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz
tar xzf boost_1_77_0.tar.gz
```

3. Build the package and install the dependencies listed in `requirements.txt`:

```sh
pip install .
```

## Usage

```python
from pcdmeshing import run_block_meshing

mesh = run_block_meshing(
        pcd: Union[Path, o3d.geometry.PointCloud],
        voxel_size: float = 20,
        margin_seam: float = 0.2,
        margin_discard: float = 0.2,
        num_parallel: int = 10,
        tmp_dir: Optional[Path] = None,
        use_visibility: bool = False,
        pcd_all_path: Optional[Path] = None,
        pcd_obs_path: Optional[Path] = None,
        opts: Dict = dict(max_edge_length=1., max_visibility=10),
) -> o3d.geometry.TriangleMesh
```
