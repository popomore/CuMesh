import torch
import numpy as np
import cumesh


def make_box_mesh():
    """Create a simple box mesh for testing."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=np.int32)
    return vertices, faces


def make_sphere_mesh(subdivisions=3):
    """Create a UV sphere mesh without external dependencies."""
    lat_steps = max(4, subdivisions * 4)
    lon_steps = max(8, subdivisions * 8)

    lat = np.linspace(0.0, np.pi, lat_steps + 1, dtype=np.float32)
    lon = np.linspace(0.0, 2.0 * np.pi, lon_steps, dtype=np.float32, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    x = np.sin(lat_grid) * np.cos(lon_grid)
    y = np.sin(lat_grid) * np.sin(lon_grid)
    z = np.cos(lat_grid)

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    faces = []
    for i in range(lat_steps):
        for j in range(lon_steps):
            j_next = (j + 1) % lon_steps
            v0 = i * lon_steps + j
            v1 = i * lon_steps + j_next
            v2 = (i + 1) * lon_steps + j
            v3 = (i + 1) * lon_steps + j_next
            if i != 0:
                faces.append([v0, v2, v1])
            if i + 1 != lat_steps:
                faces.append([v1, v2, v3])
    faces = np.asarray(faces, dtype=np.int32)
    return vertices, faces


def get_gpu_memory_used():
    """Get GPU memory used via torch.cuda.mem_get_info (tracks all CUDA allocations)."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return total - free


def test_simplify_no_leak():
    """Test that simplify() does not leak GPU memory."""
    print("\n=== Test: Simplify Memory ===")
    vertices, faces = make_sphere_mesh(4)
    print(f"Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Warmup
    mesh = cumesh.CuMesh()
    v = torch.tensor(vertices, device='cuda')
    f = torch.tensor(faces, device='cuda')
    mesh.init(v, f)
    mesh.simplify(100)
    mesh.clear_cache()
    del mesh, v, f
    torch.cuda.empty_cache()

    initial = get_gpu_memory_used()

    for _ in range(10):
        mesh = cumesh.CuMesh()
        v = torch.tensor(vertices, device='cuda')
        f = torch.tensor(faces, device='cuda')
        mesh.init(v, f)
        mesh.simplify(100)
        mesh.clear_cache()
        del mesh, v, f
        torch.cuda.empty_cache()

    final = get_gpu_memory_used()
    leak = final - initial
    print(f"Memory leak: {leak / 1024:.2f} KB")
    assert leak < 1 * 1024 * 1024, f"Potential memory leak: {leak / 1024:.2f} KB"
    print("PASSED")


def test_fill_holes_no_leak():
    """Test that fill_holes() does not leak GPU memory."""
    print("\n=== Test: Fill Holes Memory ===")
    vertices, faces = make_sphere_mesh(3)
    # Remove some faces to create holes
    faces = faces[::2]
    print(f"Mesh with holes: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Warmup
    mesh = cumesh.CuMesh()
    v = torch.tensor(vertices, device='cuda')
    f = torch.tensor(faces, device='cuda')
    mesh.init(v, f)
    mesh.fill_holes(max_hole_perimeter=1.0)
    mesh.clear_cache()
    del mesh, v, f
    torch.cuda.empty_cache()

    initial = get_gpu_memory_used()

    for _ in range(10):
        mesh = cumesh.CuMesh()
        v = torch.tensor(vertices, device='cuda')
        f = torch.tensor(faces, device='cuda')
        mesh.init(v, f)
        mesh.fill_holes(max_hole_perimeter=1.0)
        mesh.clear_cache()
        del mesh, v, f
        torch.cuda.empty_cache()

    final = get_gpu_memory_used()
    leak = final - initial
    print(f"Memory leak: {leak / 1024:.2f} KB")
    assert leak < 1 * 1024 * 1024, f"Potential memory leak: {leak / 1024:.2f} KB"
    print("PASSED")


def test_cleanup_pipeline_no_leak():
    """Test the full cleanup pipeline (remove_duplicate, repair, fill_holes, simplify)."""
    print("\n=== Test: Cleanup Pipeline Memory ===")
    vertices, faces = make_sphere_mesh(4)
    print(f"Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Warmup
    mesh = cumesh.CuMesh()
    v = torch.tensor(vertices, device='cuda')
    f = torch.tensor(faces, device='cuda')
    mesh.init(v, f)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    mesh.simplify(1000)
    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(1e-5)
    mesh.unify_face_orientations()
    mesh.clear_cache()
    del mesh, v, f
    torch.cuda.empty_cache()

    initial = get_gpu_memory_used()

    for i in range(5):
        mesh = cumesh.CuMesh()
        v = torch.tensor(vertices, device='cuda')
        f = torch.tensor(faces, device='cuda')
        mesh.init(v, f)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.simplify(1000)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.unify_face_orientations()
        mesh.clear_cache()
        del mesh, v, f
        torch.cuda.empty_cache()

        if i % 2 == 0:
            current = get_gpu_memory_used()
            print(f"Iteration {i}: {(current - initial) / 1024:+.2f} KB")

    final = get_gpu_memory_used()
    leak = final - initial
    print(f"Total leak: {leak / 1024:.2f} KB")
    assert leak < 1 * 1024 * 1024, f"Potential memory leak: {leak / 1024:.2f} KB"
    print("PASSED")


def test_clear_cache_idempotent():
    """Test that clear_cache() can be called multiple times safely."""
    print("\n=== Test: Clear Cache Idempotent ===")
    vertices, faces = make_box_mesh()

    mesh = cumesh.CuMesh()
    v = torch.tensor(vertices, device='cuda')
    f = torch.tensor(faces, device='cuda')
    mesh.init(v, f)
    mesh.simplify(6)
    mesh.clear_cache()
    mesh.clear_cache()
    mesh.clear_cache()  # Should not crash
    del mesh
    print("PASSED")


if __name__ == '__main__':
    print("=" * 60)
    print("CuMesh Memory Management Tests")
    print("=" * 60)

    test_clear_cache_idempotent()
    test_simplify_no_leak()
    test_fill_holes_no_leak()
    test_cleanup_pipeline_no_leak()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
