#pragma once

#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Filtered_kernel.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>

#include <CGAL/Point_set_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/property_map.h>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>

//typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
// Define a kernel fp32 geometric primitive instead of the default fp64
class Kernel
  : public CGAL::Filtered_kernel_adaptor<
        CGAL::Type_equality_wrapper<
            CGAL::Simple_cartesian<float>::Base<Kernel>::Type, Kernel>>
{};

// Basic primitives
typedef Kernel::Point_3 Point_3;
typedef Kernel::Vector_3 Vector_3;
typedef Kernel::Segment_3 Segment;
typedef CGAL::Point_set_3<Point_3> Point_set;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef std::array<std::size_t, 3> Triangle;
typedef std::array<unsigned char, 3> Color;

// Point with normal, color and intensity
typedef std::tuple<Point_3, Vector_3, Color> PNC;
typedef CGAL::Nth_of_tuple_property_map<0, PNC> Point_map;
typedef CGAL::Nth_of_tuple_property_map<1, PNC> Normal_map;
typedef CGAL::Nth_of_tuple_property_map<2, PNC> Color_map;
typedef std::vector<PNC> ColoredPoints;

// Reconstruction types
typedef CGAL::Advancing_front_surface_reconstruction_vertex_base_3<Kernel> LVb;
typedef CGAL::Advancing_front_surface_reconstruction_cell_base_3<Kernel> LCb;
typedef CGAL::Triangulation_data_structure_3<LVb, LCb> Tds;
typedef CGAL::Delaunay_triangulation_3<Kernel, Tds, CGAL::Fast_location> Triangulation_3;
typedef std::unordered_map<
  Triangulation_3::Facet, int, boost::hash<Triangulation_3::Facet>> VisibilityCounter;

// Ray-mesh intersection
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive> AABBTraits;
typedef CGAL::AABB_tree<AABBTraits> AABBTree;
