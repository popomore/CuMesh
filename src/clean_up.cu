#include "cumesh.h"
#include "dtypes.cuh"
#include "shared.h"
#include <cub/cub.cuh>


namespace cumesh {


static __global__ void copy_vec3f_to_float3_kernel(
    const Vec3f* vec3f,
    const size_t N,
    float3* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[tid] = make_float3(vec3f[tid].x, vec3f[tid].y, vec3f[tid].z);
}


template<typename T, typename U>
static __global__ void copy_T_to_T3_kernel(
    const T* input,
    const size_t N,
    U* output
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[tid] = { input[3 * tid], input[3 * tid + 1], input[3 * tid + 2] };
}


void CuMesh::remove_faces(torch::Tensor& face_mask) {
    size_t F = this->faces.size;

    size_t temp_storage_bytes = 0;
    CudaPtr<int> cu_new_num_faces(1);
    CudaPtr<int3> cu_new_faces(F);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        this->faces.ptr, face_mask.data_ptr<bool>(), cu_new_faces.get(), cu_new_num_faces.get(),
        F
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->faces.ptr, face_mask.data_ptr<bool>(), cu_new_faces.get(), cu_new_num_faces.get(),
        F
    ));
    int new_num_faces;
    CUDA_CHECK(cudaMemcpy(&new_num_faces, cu_new_num_faces.get(), sizeof(int), cudaMemcpyDeviceToHost));
    this->faces.resize(new_num_faces);
    CUDA_CHECK(cudaMemcpy(this->faces.ptr, cu_new_faces.get(), new_num_faces * sizeof(int3), cudaMemcpyDeviceToDevice));

    this->remove_unreferenced_vertices();
}


void CuMesh::_remove_faces(uint8_t* face_mask) {
    size_t F = this->faces.size;

    size_t temp_storage_bytes = 0;
    CudaPtr<int> cu_new_num_faces(1);
    CudaPtr<int3> cu_new_faces(F);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        this->faces.ptr, face_mask, cu_new_faces.get(), cu_new_num_faces.get(),
        F
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->faces.ptr, face_mask, cu_new_faces.get(), cu_new_num_faces.get(),
        F
    ));
    int new_num_faces;
    CUDA_CHECK(cudaMemcpy(&new_num_faces, cu_new_num_faces.get(), sizeof(int), cudaMemcpyDeviceToHost));
    this->faces.resize(new_num_faces);
    CUDA_CHECK(cudaMemcpy(this->faces.ptr, cu_new_faces.get(), new_num_faces * sizeof(int3), cudaMemcpyDeviceToDevice));

    this->remove_unreferenced_vertices();
}


static __global__ void set_vertex_is_referenced(
    const int3* faces,
    const size_t F,
    int* vertex_is_referenced
) {
    const int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= F) return;
    int3 face = faces[fid];
    vertex_is_referenced[face.x] = 1;
    vertex_is_referenced[face.y] = 1;
    vertex_is_referenced[face.z] = 1;
}


static __global__ void compress_vertices_kernel(
    const int* vertices_map,
    const float3* old_vertices,
    const int V,
    float3* new_vertices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    int new_id = vertices_map[tid];
    int is_kept = vertices_map[tid + 1] == new_id + 1;
    if (is_kept) {
        new_vertices[new_id] = old_vertices[tid];
    }
}


static __global__ void remap_faces_kernel(
    const int* vertices_map,
    const int F,
    int3* faces
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    faces[tid].x = vertices_map[faces[tid].x];
    faces[tid].y = vertices_map[faces[tid].y];
    faces[tid].z = vertices_map[faces[tid].z];
}


void CuMesh::remove_unreferenced_vertices() {
    size_t V = this->vertices.size;
    size_t F = this->faces.size;

    // Mark referenced vertices
    CudaPtr<int> cu_vertex_is_referenced(V + 1);
    CUDA_CHECK(cudaMemset(cu_vertex_is_referenced.get(), 0, (V+1) * sizeof(int)));
    set_vertex_is_referenced<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->faces.ptr,
        F,
        cu_vertex_is_referenced.get()
    );
    CUDA_CHECK(cudaGetLastError());

    // Get vertices map
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_vertex_is_referenced.get(), V+1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_vertex_is_referenced.get(), V+1
    ));
    int new_num_vertices;
    CUDA_CHECK(cudaMemcpy(&new_num_vertices, cu_vertex_is_referenced.get() + V, sizeof(int), cudaMemcpyDeviceToHost));

    // Compress vertices
    this->temp_storage.resize(new_num_vertices * sizeof(float3));
    compress_vertices_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_vertex_is_referenced.get(),
        this->vertices.ptr,
        V,
        reinterpret_cast<float3*>(this->temp_storage.ptr)
    );
    CUDA_CHECK(cudaGetLastError());
    swap_buffers(this->temp_storage, this->vertices);

    // Update faces
    remap_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_vertex_is_referenced.get(),
        F,
        this->faces.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    // Delete all cached info since mesh has changed
    this->clear_cache();
}


static __global__ void sort_faces_kernel(
    int3* faces,
    const size_t F
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int3 face = faces[tid];
    int tmp;

    // bubble sort 3 elements (x, y, z)
    if (face.x > face.y) { tmp = face.x; face.x = face.y; face.y = tmp; }
    if (face.y > face.z) { tmp = face.y; face.y = face.z; face.z = tmp; }
    if (face.x > face.y) { tmp = face.x; face.x = face.y; face.y = tmp; }

    faces[tid] = face;
}


static __global__ void select_first_in_each_group_kernel(
    const int3* faces,
    const size_t F,
    uint8_t* face_mask
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    if (tid == 0) {
        face_mask[tid] = 1;
    } else {
        int3 face = faces[tid];
        int3 prev_face = faces[tid-1];
        if (face.x == prev_face.x && face.y == prev_face.y && face.z == prev_face.z) {
            face_mask[tid] = 0;
        } else {
            face_mask[tid] = 1;
        }
    }
}


struct int3_decomposer
{
    __host__ __device__ ::cuda::std::tuple<int&, int&, int&> operator()(int3& key) const
    {
        return {key.x, key.y, key.z};
    }
};


void CuMesh::remove_duplicate_faces() {
    size_t F = this->faces.size;

    // Create a temporary sorted copy of faces for duplicate detection
    // Do NOT modify the original faces to preserve vertex order and normals
    CudaPtr<int3> cu_sorted_faces(F);
    CUDA_CHECK(cudaMemcpy(cu_sorted_faces.get(), this->faces.ptr, F * sizeof(int3), cudaMemcpyDeviceToDevice));

    // Sort vertices within each face (in the temporary copy)
    sort_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_sorted_faces.get(),
        F
    );
    CUDA_CHECK(cudaGetLastError());

    // Sort all faces globally by their sorted vertex indices
    size_t temp_storage_bytes = 0;
    CudaPtr<int> cu_sorted_face_indices(F);
    arange_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(cu_sorted_face_indices.get(), F);
    CUDA_CHECK(cudaGetLastError());

    CudaPtr<int> cu_sorted_indices_output(F);
    CudaPtr<int3> cu_sorted_faces_output(F);

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        cu_sorted_faces.get(), cu_sorted_faces_output.get(),
        cu_sorted_face_indices.get(), cu_sorted_indices_output.get(),
        F,
        int3_decomposer{}
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_sorted_faces.get(), cu_sorted_faces_output.get(),
        cu_sorted_face_indices.get(), cu_sorted_indices_output.get(),
        F,
        int3_decomposer{}
    ));

    // Select first in each group of duplicate faces (based on sorted faces)
    CudaPtr<uint8_t> cu_face_mask_sorted(F);
    select_first_in_each_group_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_sorted_faces_output.get(),
        F,
        cu_face_mask_sorted.get()
    );
    CUDA_CHECK(cudaGetLastError());

    // Map the mask back to original face order using scatter
    // scatter: output[indices[i]] = values[i]
    // This maps: cu_face_mask_original[original_idx] = cu_face_mask_sorted[sorted_position]
    CudaPtr<uint8_t> cu_face_mask_original(F);
    scatter_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_sorted_indices_output.get(),  // indices: sorted_position -> original_idx
        cu_face_mask_sorted.get(),       // values: mask at sorted_position
        F,
        cu_face_mask_original.get()      // output: mask at original position
    );
    CUDA_CHECK(cudaGetLastError());

    // Select faces to keep (preserving original vertex order)
    this->_remove_faces(cu_face_mask_original.get());
}


static __global__ void mark_degenerate_faces_kernel(
    const float3* vertices,
    const int3* faces,
    const float abs_thresh,
    const float rel_thresh,
    const size_t F,
    uint8_t* face_mask
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    int3 face = faces[tid];

    // 1. Check if any vertex is duplicated
    if (face.x == face.y || face.y == face.z || face.z == face.x) {
        face_mask[tid] = 0;
        return;
    }

    // 2. Check if slim or zero area
    Vec3f v0 = Vec3f(vertices[face.x]);
    Vec3f v1 = Vec3f(vertices[face.y]);
    Vec3f v2 = Vec3f(vertices[face.z]);
    Vec3f e0 = v1 - v0;
    Vec3f e1 = v2 - v1;
    Vec3f e2 = v0 - v2;
    float max_edge_len = fmaxf(fmaxf(e0.norm(), e1.norm()), e2.norm());
    float area = e0.cross(e1).norm() / 2.0f;
    float thresh = fminf(rel_thresh * max_edge_len * max_edge_len, abs_thresh);
    if (area < thresh) {
        face_mask[tid] = 0;
        return;
    }

    face_mask[tid] = 1;
}


void CuMesh::remove_degenerate_faces(float abs_thresh, float rel_thresh) {
    size_t F = this->faces.size;

    CudaPtr<uint8_t> cu_face_mask(F);
    mark_degenerate_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->vertices.ptr,
        this->faces.ptr,
        abs_thresh, rel_thresh,
        F,
        cu_face_mask.get()
    );
    CUDA_CHECK(cudaGetLastError());

    this->_remove_faces(cu_face_mask.get());
}


static __global__ void compute_loop_boundary_lengths(
    const float3* vertices,
    const uint64_t* edges,
    const int* loop_boundaries,
    const size_t E,
    float* loop_boundary_lengths
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;
    uint64_t edge = edges[loop_boundaries[tid]];
    int e0 = int(edge & 0xFFFFFFFF);
    int e1 = int(edge >> 32);
    Vec3f v0 = Vec3f(vertices[e0]);
    Vec3f v1 = Vec3f(vertices[e1]);
    loop_boundary_lengths[tid] = (v1 - v0).norm();
}


static __global__ void compute_loop_boundary_midpoints(
    const float3* vertices,
    const uint64_t* edges,
    const int* loop_boundaries,
    const size_t E,
    Vec3f* loop_boundary_midpoints
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;
    uint64_t edge = edges[loop_boundaries[tid]];
    int e0 = int(edge & 0xFFFFFFFF);
    int e1 = int(edge >> 32);
    Vec3f v0 = Vec3f(vertices[e0]);
    Vec3f v1 = Vec3f(vertices[e1]);
    loop_boundary_midpoints[tid] = (v0 + v1) * 0.5f;
}


static __global__ void connect_new_vertices_kernel(
    const uint64_t* edges,
    const int* loop_boundaries,
    const int* loop_bound_loop_ids,
    const size_t L,
    const size_t V,
    int3* faces
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= L) return;
    int loop_id = loop_bound_loop_ids[tid];
    int loop_boundary = loop_boundaries[tid];
    uint64_t e = edges[loop_boundary];
    int e0 = int(e & 0xFFFFFFFF);
    int e1 = int(e >> 32);
    int new_v_id = loop_id + V;
    faces[tid] = {e0, e1, new_v_id};
}


struct LessThanOp {
    __device__ bool operator()(float a, float b) const {
        return a < b;
    }
};


void CuMesh::fill_holes(float max_hole_perimeter) {
    if (this->loop_boundaries.is_empty() || this->loop_boundaries_offset.is_empty()) {
        this->get_boundary_loops();
    }

    size_t V = this->vertices.size;
    size_t F = this->faces.size;
    size_t L = this->num_bound_loops;
    size_t E = this->loop_boundaries.size;

    // Early return if no boundary loops
    if (L == 0 || E == 0) {
        return;
    }

    // Compute loop boundary lengths
    CudaPtr<float> cu_loop_boundary_lengths(E);
    compute_loop_boundary_lengths<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->vertices.ptr,
        this->edges.ptr,
        this->loop_boundaries.ptr,
        E,
        cu_loop_boundary_lengths.get()
    );
    CUDA_CHECK(cudaGetLastError());

    // Segment sum
    size_t temp_storage_bytes = 0;
    CudaPtr<float> cu_bound_loop_perimeters(L);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_bytes,
        cu_loop_boundary_lengths.get(), cu_bound_loop_perimeters.get(),
        L,
        this->loop_boundaries_offset.ptr,
        this->loop_boundaries_offset.ptr + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_loop_boundary_lengths.get(), cu_bound_loop_perimeters.get(),
        L,
        this->loop_boundaries_offset.ptr,
        this->loop_boundaries_offset.ptr + 1
    ));

    // Mask small loops
    CudaPtr<uint8_t> cu_bound_loop_mask(L);
    compare_kernel<<<(L+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_bound_loop_perimeters.get(),
        max_hole_perimeter,
        L,
        LessThanOp(),
        cu_bound_loop_mask.get()
    );
    CUDA_CHECK(cudaGetLastError());

    // Compress bound loops size
    CudaPtr<int> cu_bound_loops_cnt(L);
    diff_kernel<<<(L+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->loop_boundaries_offset.ptr,
        L,
        cu_bound_loops_cnt.get()
    );
    CUDA_CHECK(cudaGetLastError());
    CudaPtr<int> cu_new_loop_boundaries_cnt(L + 1);
    CudaPtr<int> cu_new_num_bound_loops(1);
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        cu_bound_loops_cnt.get(), cu_bound_loop_mask.get(), cu_new_loop_boundaries_cnt.get(), cu_new_num_bound_loops.get(),
        L
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_bound_loops_cnt.get(), cu_bound_loop_mask.get(), cu_new_loop_boundaries_cnt.get(), cu_new_num_bound_loops.get(),
        L
    ));
    int new_num_bound_loops;
    CUDA_CHECK(cudaMemcpy(&new_num_bound_loops, cu_new_num_bound_loops.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (new_num_bound_loops == 0) {
        return;
    }

    // Get loop ids of loop boundaries
    CudaPtr<int> cu_loop_bound_loop_ids(E);
    CUDA_CHECK(cudaMemset(cu_loop_bound_loop_ids.get(), 0, E * sizeof(int)));
    if (L > 1) {
        set_flag_kernel<<<(L-1+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            this->loop_boundaries_offset.ptr + 1, L - 1,
            cu_loop_bound_loop_ids.get()
        );
        CUDA_CHECK(cudaGetLastError());
    }
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes,
        cu_loop_bound_loop_ids.get(),
        E
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_loop_bound_loop_ids.get(),
        E
    ));

    // Mask loop boundaries
    CudaPtr<uint8_t> cu_loop_boundary_mask(E);
    index_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_bound_loop_mask.get(),
        cu_loop_bound_loop_ids.get(),
        E,
        cu_loop_boundary_mask.get()
    );
    CUDA_CHECK(cudaGetLastError());

    // Compress loop boundaries
    CudaPtr<int> cu_new_loop_boundaries(E);
    CudaPtr<int> cu_new_num_loop_boundaries(1);
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        this->loop_boundaries.ptr, cu_loop_boundary_mask.get(), cu_new_loop_boundaries.get(), cu_new_num_loop_boundaries.get(),
        E
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->loop_boundaries.ptr, cu_loop_boundary_mask.get(), cu_new_loop_boundaries.get(), cu_new_num_loop_boundaries.get(),
        E
    ));
    int new_num_loop_boundaries;
    CUDA_CHECK(cudaMemcpy(&new_num_loop_boundaries, cu_new_num_loop_boundaries.get(), sizeof(int), cudaMemcpyDeviceToHost));

    // Reconstruct new bound loops
    CudaPtr<int> cu_new_loop_boundaries_offset(new_num_loop_boundaries + 1);
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_new_loop_boundaries_cnt.get(), cu_new_loop_boundaries_offset.get(),
        new_num_bound_loops + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_new_loop_boundaries_cnt.get(), cu_new_loop_boundaries_offset.get(),
        new_num_bound_loops + 1
    ));
    CudaPtr<int> cu_new_loop_bound_loop_ids(new_num_loop_boundaries);
    CUDA_CHECK(cudaMemset(cu_new_loop_bound_loop_ids.get(), 0, new_num_loop_boundaries * sizeof(int)));
    if (new_num_bound_loops > 1) {
        set_flag_kernel<<<(new_num_bound_loops-1+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            cu_new_loop_boundaries_offset.get()+1, new_num_bound_loops-1,
            cu_new_loop_bound_loop_ids.get()
        );
        CUDA_CHECK(cudaGetLastError());
    }
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes,
        cu_new_loop_bound_loop_ids.get(),
        new_num_loop_boundaries
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_new_loop_bound_loop_ids.get(),
        new_num_loop_boundaries
    ));

    // Calculate new vertex positions as average of loop vertices
    CudaPtr<Vec3f> cu_new_loop_bound_centers(new_num_loop_boundaries);
    compute_loop_boundary_midpoints<<<(new_num_loop_boundaries+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->vertices.ptr,
        this->edges.ptr,
        cu_new_loop_boundaries.get(),
        new_num_loop_boundaries,
        cu_new_loop_bound_centers.get()
    );
    CUDA_CHECK(cudaGetLastError());
    CudaPtr<Vec3f> cu_new_vertices(new_num_bound_loops);
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_bytes,
        cu_new_loop_bound_centers.get(), cu_new_vertices.get(),
        new_num_bound_loops,
        cu_new_loop_boundaries_offset.get(),
        cu_new_loop_boundaries_offset.get() + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_new_loop_bound_centers.get(), cu_new_vertices.get(),
        new_num_bound_loops,
        cu_new_loop_boundaries_offset.get(),
        cu_new_loop_boundaries_offset.get() + 1
    ));
    inplace_div_kernel<<<(new_num_bound_loops+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_new_vertices.get(),
        cu_new_loop_boundaries_cnt.get(),
        new_num_bound_loops
    );
    CUDA_CHECK(cudaGetLastError());

    // Update mesh
    this->vertices.extend(new_num_bound_loops);
    this->faces.extend(new_num_loop_boundaries);
    copy_vec3f_to_float3_kernel<<<(new_num_bound_loops+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_new_vertices.get(),
        new_num_bound_loops,
        this->vertices.ptr + V
    );
    CUDA_CHECK(cudaGetLastError());
    connect_new_vertices_kernel<<<(new_num_loop_boundaries+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->edges.ptr,
        cu_new_loop_boundaries.get(),
        cu_new_loop_bound_loop_ids.get(),
        new_num_loop_boundaries,
        V,
        this->faces.ptr + F
    );
    CUDA_CHECK(cudaGetLastError());

    // Delete all cached info since mesh has changed
    this->clear_cache();
}


static __global__ void construct_vertex_adj_pairs_kernel(
    const int2* manifold_face_adj,
    const int3* faces,
    int2* vertex_adj_pairs,
    const size_t M
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    const int2 adj_faces = manifold_face_adj[tid];
    const int3 face1 = faces[adj_faces.x];
    const int3 face2 = faces[adj_faces.y];

    const int v1[3] = {face1.x, face1.y, face1.z};

    int shared_local_indices1[2] = {0, 0};
    int shared_local_indices2[2] = {0, 0};
    int found_count = 0;

    for (int i = 0; i < 3; ++i) {
        if (v1[i] == face2.x) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 0;
            found_count++;
        } else if (v1[i] == face2.y) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 1;
            found_count++;
        } else if (v1[i] == face2.z) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 2;
            found_count++;
        }
        if (found_count == 2) {
            break;
        }
    }

    // Only process if we found exactly 2 shared vertices (valid manifold edge)
    if (found_count == 2) {
        vertex_adj_pairs[2 * tid + 0] = make_int2(
            3 * adj_faces.x + shared_local_indices1[0],
            3 * adj_faces.y + shared_local_indices2[0]
        );
        vertex_adj_pairs[2 * tid + 1] = make_int2(
            3 * adj_faces.x + shared_local_indices1[1],
            3 * adj_faces.y + shared_local_indices2[1]
        );
    } else {
        // Invalid edge, set to identity mapping
        vertex_adj_pairs[2 * tid + 0] = make_int2(3 * adj_faces.x, 3 * adj_faces.x);
        vertex_adj_pairs[2 * tid + 1] = make_int2(3 * adj_faces.y, 3 * adj_faces.y);
    }
}


static __global__ void index_vertice_kernel(
    const int* vertex_ids,
    const int3* faces,
    const float3* vertices,
    const size_t V,
    float3* new_vertices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    const int vid = vertex_ids[tid];
    const int3 face = faces[vid / 3];
    const int f[3] = {face.x, face.y, face.z};
    new_vertices[tid] = vertices[f[vid % 3]];
}


void CuMesh::repair_non_manifold_edges(){
    // Always recompute manifold_face_adj to ensure it's up to date
    // especially after operations like simplify() that modify the mesh
    this->get_manifold_face_adjacency();

    size_t F = this->faces.size;
    size_t M = this->manifold_face_adj.size;

    // Construct vertex adjacency pairs with manifold edges
    CudaPtr<int2> cu_vertex_adj_pairs(2 * M);
    construct_vertex_adj_pairs_kernel<<<(M+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->manifold_face_adj.ptr,
        this->faces.ptr,
        cu_vertex_adj_pairs.get(),
        M
    );
    CUDA_CHECK(cudaGetLastError());

    // Iterative Hook and Compress
    CudaPtr<int> cu_vertex_ids(3 * F);
    arange_kernel<<<(3*F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(cu_vertex_ids.get(), 3 * F);
    CUDA_CHECK(cudaGetLastError());
    CudaPtr<int> cu_end_flag(1);
    int h_end_flag;
    do {
        h_end_flag = 1;
        CUDA_CHECK(cudaMemcpy(cu_end_flag.get(), &h_end_flag, sizeof(int), cudaMemcpyHostToDevice));

        // Hook
        hook_edges_kernel<<<(2*M+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            cu_vertex_adj_pairs.get(),
            2 * M,
            cu_vertex_ids.get(),
            cu_end_flag.get()
        );
        CUDA_CHECK(cudaGetLastError());

        // Compress
        compress_components_kernel<<<(3*F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            cu_vertex_ids.get(),
            3 * F
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_end_flag, cu_end_flag.get(), sizeof(int), cudaMemcpyDeviceToHost));
    } while (h_end_flag == 0);

    // Construct new faces
    CudaPtr<int> cu_new_vertices_ids(3 * F);
    int new_V = compress_ids(cu_vertex_ids.get(), 3 * F, this->cub_temp_storage, cu_new_vertices_ids.get());
    CudaPtr<float3> cu_new_vertices(new_V);
    index_vertice_kernel<<<(new_V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_new_vertices_ids.get(),
        this->faces.ptr,
        this->vertices.ptr,
        new_V,
        cu_new_vertices.get()
    );
    CUDA_CHECK(cudaGetLastError());
    this->vertices.resize(new_V);
    CUDA_CHECK(cudaMemcpy(this->vertices.ptr, cu_new_vertices.get(), new_V * sizeof(float3), cudaMemcpyDeviceToDevice));
    this->faces.resize(F);
    copy_T_to_T3_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(cu_vertex_ids.get(), F, this->faces.ptr);
    CUDA_CHECK(cudaGetLastError());

    // Delete all cached info since mesh has changed
    this->clear_cache();
}


/**
 * Mark faces to remove for non-manifold edges
 * For each non-manifold edge (shared by >2 faces), only keep the first 2 faces
 *
 * @param edge2face: edge to face adjacency
 * @param edge2face_offset: edge to face adjacency offset
 * @param edge2face_cnt: number of faces per edge
 * @param E: number of edges
 * @param face_keep_mask: output mask (1 = keep, 0 = remove)
 */
static __global__ void mark_non_manifold_faces_kernel(
    const int* edge2face,
    const int* edge2face_offset,
    const int* edge2face_cnt,
    const size_t E,
    uint8_t* face_keep_mask
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // Only process non-manifold edges (cnt > 2)
    int cnt = edge2face_cnt[tid];
    if (cnt <= 2) return;

    // Mark faces beyond the first 2 for removal
    int start = edge2face_offset[tid];
    for (int i = 2; i < cnt; i++) {
        int face_idx = edge2face[start + i];
        face_keep_mask[face_idx] = 0;
    }
}


void CuMesh::remove_non_manifold_faces() {
    // Get edge-face adjacency information
    if (this->edge2face.is_empty() || this->edge2face_offset.is_empty()) {
        this->get_edge_face_adjacency();
    }

    size_t F = this->faces.size;
    size_t E = this->edges.size;

    if (F == 0 || E == 0) return;

    // Initialize face mask (1 = keep all faces initially)
    CudaPtr<uint8_t> cu_face_keep_mask(F);
    CUDA_CHECK(cudaMemset(cu_face_keep_mask.get(), 1, F * sizeof(uint8_t)));

    // Mark faces on non-manifold edges for removal
    mark_non_manifold_faces_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->edge2face.ptr,
        this->edge2face_offset.ptr,
        this->edge2face_cnt.ptr,
        E,
        cu_face_keep_mask.get()
    );
    CUDA_CHECK(cudaGetLastError());

    // Remove marked faces
    this->_remove_faces(cu_face_keep_mask.get());

    // Clear cache since mesh has changed
    this->clear_cache();
}


struct GreaterThanOrEqualToOp {
    __device__ __forceinline__ bool operator()(const float& a, const float& b) const {
        return a >= b;
    }
};


void CuMesh::remove_small_connected_components(float min_area) {
    if (this->conn_comp_ids.is_empty()) {
        this->get_connected_components();
    }
    if (this->face_areas.is_empty()) {
        this->compute_face_areas();
    }
    size_t F = this->faces.size;
    if (F == 0) return;

    // 1. Sort face areas based on their connected component ID.
    // This groups all faces of the same component together.
    size_t temp_storage_bytes = 0;
    CudaPtr<int> cu_sorted_conn_comp_ids(F);
    CudaPtr<float> cu_sorted_face_areas(F);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        this->conn_comp_ids.ptr, cu_sorted_conn_comp_ids.get(),
        this->face_areas.ptr, cu_sorted_face_areas.get(),
        F
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->conn_comp_ids.ptr, cu_sorted_conn_comp_ids.get(),
        this->face_areas.ptr, cu_sorted_face_areas.get(),
        F
    ));

    // 2. Find unique components and get the number of faces in each.
    CudaPtr<int> cu_conn_comp_num_faces(this->num_conn_comps + 1);
    CudaPtr<int> cu_num_conn_comps(1);
    CudaPtr<int> cu_unique_conn_comp_ids(this->num_conn_comps + 1);  // Not needed, but we need to pass a valid pointer.
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr, temp_storage_bytes,
        cu_sorted_conn_comp_ids.get(), cu_unique_conn_comp_ids.get(),
        cu_conn_comp_num_faces.get(), cu_num_conn_comps.get(),
        F
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_sorted_conn_comp_ids.get(), cu_unique_conn_comp_ids.get(),
        cu_conn_comp_num_faces.get(), cu_num_conn_comps.get(),
        F
    ));
    int num_conn_comps;
    CUDA_CHECK(cudaMemcpy(&num_conn_comps, cu_num_conn_comps.get(), sizeof(int), cudaMemcpyDeviceToHost));

    // 3. Compute the total area for each connected component via segmented reduction.
    CudaPtr<int> cu_conn_comp_offsets(num_conn_comps + 1);
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        cu_conn_comp_num_faces.get(), cu_conn_comp_offsets.get(),
        num_conn_comps + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_conn_comp_num_faces.get(), cu_conn_comp_offsets.get(),
        num_conn_comps + 1
    ));

    CudaPtr<float> cu_conn_comp_areas(num_conn_comps);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_bytes,
        cu_sorted_face_areas.get(), cu_conn_comp_areas.get(),
        num_conn_comps,
        cu_conn_comp_offsets.get(),
        cu_conn_comp_offsets.get() + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_sorted_face_areas.get(), cu_conn_comp_areas.get(),
        num_conn_comps,
        cu_conn_comp_offsets.get(),
        cu_conn_comp_offsets.get() + 1
    ));

    // 4. Create a "keep" mask for components with area >= min_area.
    CudaPtr<uint8_t> cu_comp_keep_mask(num_conn_comps);
    compare_kernel<<<(num_conn_comps+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_conn_comp_areas.get(),
        min_area,
        num_conn_comps,
        GreaterThanOrEqualToOp(),
        cu_comp_keep_mask.get()
    );
    CUDA_CHECK(cudaGetLastError());

    // 5. Propagate the component "keep" mask to every face.
    CudaPtr<uint8_t> cu_face_keep_mask(F);
    // Use an index_kernel (gather operation)
    index_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_comp_keep_mask.get(),  // Source array
        this->conn_comp_ids.ptr,  // Indices to gather from
        F,
        cu_face_keep_mask.get()   // Destination array
    );
    CUDA_CHECK(cudaGetLastError());

    // 6. Select the faces to keep and update the mesh.
    this->_remove_faces(cu_face_keep_mask.get());
}


static __global__ void hook_edges_with_orientation_kernel(
    const int2* adj,
    const uint8_t* flipped,
    const int M,
    int* conn_comp_ids,
    int* end_flag
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    // get adjacent faces
    int f0 = adj[tid].x;
    int f1 = adj[tid].y;
    uint8_t is_flipped = flipped[tid];

    // union
    // find roots
    int root0 = conn_comp_ids[f0] >> 1;
    int flip0 = conn_comp_ids[f0] & 1;
    while (root0 != (conn_comp_ids[root0] >> 1)) {
        flip0 ^= conn_comp_ids[root0] & 1;
        root0 = conn_comp_ids[root0] >> 1;
    }
    int root1 = conn_comp_ids[f1] >> 1;
    int flip1 = conn_comp_ids[f1] & 1;
    while (root1 != (conn_comp_ids[root1] >> 1)) {
        flip1 ^= conn_comp_ids[root1] & 1;
        root1 = conn_comp_ids[root1] >> 1;
    }

    if (root0 == root1) return;

    int high = max(root0, root1);
    int low = min(root0, root1);
    atomicMin(&conn_comp_ids[high], (low << 1) | (is_flipped ^ flip0 ^ flip1));
    *end_flag = 0;
}


static __global__ void compress_components_with_orientation_kernel(
    int* conn_comp_ids,
    const int F
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int p = conn_comp_ids[tid] >> 1;
    int f = conn_comp_ids[tid] & 1;
    while (p != (conn_comp_ids[p] >> 1)) {
        f ^= conn_comp_ids[p] & 1;
        p = conn_comp_ids[p] >> 1;
    }
    conn_comp_ids[tid] = (p << 1) | f;
}


static __global__ void get_flip_flags_kernel(
    const int2* manifold_face_adj,
    const int3* faces,
    const int M,
    uint8_t* flipped
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    const int2 adj_faces = manifold_face_adj[tid];
    const int3 face1 = faces[adj_faces.x];
    const int3 face2 = faces[adj_faces.y];

    const int v1[3] = {face1.x, face1.y, face1.z};

    int shared_local_indices1[2];
    int shared_local_indices2[2];
    int found_count = 0;

    for (int i = 0; i < 3; ++i) {
        if (v1[i] == face2.x) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 0;
            found_count++;
        } else if (v1[i] == face2.y) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 1;
            found_count++;
        } else if (v1[i] == face2.z) {
            shared_local_indices1[found_count] = i;
            shared_local_indices2[found_count] = 2;
            found_count++;
        }
        if (found_count == 2) {
            break;
        }
    }

    int direction1 = (shared_local_indices1[1] - shared_local_indices1[0] + 3) % 3;
    int direction2 = (shared_local_indices2[1] - shared_local_indices2[0] + 3) % 3;
    flipped[tid] = (direction1 == direction2) ? 1 : 0;
}


static __global__ void inplace_flip_faces_with_flags_kernel(
    int3* faces,
    const int* conn_comp_with_flip,
    const int F
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int is_flipped = conn_comp_with_flip[tid] & 1;
    if (is_flipped) {
        int3 face = faces[tid];
        faces[tid] = make_int3(face.x, face.z, face.y);
    }
}


void CuMesh::unify_face_orientations() {
    if (this->manifold_face_adj.is_empty()) {
        this->get_manifold_face_adjacency();
    }

    // 1. Compute the flipped flag for each edge.
    CudaPtr<uint8_t> cu_flipped(this->manifold_face_adj.size);
    get_flip_flags_kernel<<<(this->manifold_face_adj.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->manifold_face_adj.ptr,
        this->faces.ptr,
        this->manifold_face_adj.size,
        cu_flipped.get()
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. Hook edges with flipped flag.
    CudaPtr<int> conn_comp_with_flip(this->faces.size);
    arange_kernel<<<(this->faces.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(conn_comp_with_flip.get(), this->faces.size, 2);
    CUDA_CHECK(cudaGetLastError());
    CudaPtr<int> cu_end_flag(1);
    int h_end_flag;
    do {
        h_end_flag = 1;
        CUDA_CHECK(cudaMemcpy(cu_end_flag.get(), &h_end_flag, sizeof(int), cudaMemcpyHostToDevice));

        // Hook
        hook_edges_with_orientation_kernel<<<(this->manifold_face_adj.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            this->manifold_face_adj.ptr,
            cu_flipped.get(),
            this->manifold_face_adj.size,
            conn_comp_with_flip.get(),
            cu_end_flag.get()
        );
        CUDA_CHECK(cudaGetLastError());

        // Compress
        compress_components_with_orientation_kernel<<<(this->faces.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            conn_comp_with_flip.get(),
            this->faces.size
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_end_flag, cu_end_flag.get(), sizeof(int), cudaMemcpyDeviceToHost));
    } while (h_end_flag == 0);

    // 3. Flip the orientation of the faces.
    inplace_flip_faces_with_flags_kernel<<<(this->faces.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->faces.ptr,
        conn_comp_with_flip.get(),
        this->faces.size
    );
    CUDA_CHECK(cudaGetLastError());
}


} // namespace cumesh