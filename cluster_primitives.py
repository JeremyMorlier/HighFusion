"""
Gluon DSL for ClusterFusion Primitives
A Triton-like Python DSL for cluster-level collective operations
"""

import triton
import triton.language as tl


class ClusterGroup:
    """Abstraction for CUDA cluster group operations"""

    @staticmethod
    @triton.jit
    def num_blocks():
        """Returns the number of blocks in the cluster"""
        return tl.num_programs(axis=0)

    @staticmethod
    @triton.jit
    def block_rank():
        """Returns the rank of current block in cluster"""
        return tl.program_id(axis=0)

    @staticmethod
    @triton.jit
    def sync():
        """Synchronize all blocks in cluster"""
        tl.debug_barrier()


# ===== Cluster-Gather Primitive =====
@triton.jit
def cluster_gather(
    src_ptr,  # Source data pointer
    dst_ptr,  # Destination buffer pointer
    size: tl.constexpr,  # Size of data to transfer
    tile_size: tl.constexpr,  # Tile size for processing
    CLUSTER_SIZE: tl.constexpr,  # Number of blocks in cluster
):
    """
    Cluster-Gather: Gather data from source to all destination blocks

    This implements the data distribution phase where each block's
    data is copied to all other blocks in the cluster.

    Args:
        src_ptr:  Pointer to source data in shared memory
        dst_ptr:  Pointer to destination buffer in shared memory
        size: Number of bytes to transfer
        tile_size:  Size of each processing tile
        CLUSTER_SIZE:  Number of blocks in the cluster
    """
    cluster = ClusterGroup()
    cluster_block_id = cluster.block_rank()
    tid = tl.program_id(axis=1)

    # Iterate over all other blocks in cluster
    for i in range(1, CLUSTER_SIZE - 1):
        # Calculate destination block in round-robin fashion
        dst_block = (cluster_block_id + i) % CLUSTER_SIZE

        # Map address to neighbor block's shared memory
        # In Triton, we simulate this with offset calculation
        neighbor_offset = dst_block * tile_size

        # Leader thread performs async copy
        if tid == 0:
            # Bulk copy to neighbor (simulated with tl.store)
            offsets = tl.arange(0, size)
            src_data = tl.load(src_ptr + offsets)
            tl.store(dst_ptr + neighbor_offset + offsets, src_data)

        # Synchronize cluster
        cluster.sync()


# ===== Cluster-Reduce Primitive =====
@triton.jit
def cluster_reduce(
    src_ptr,  # Source data pointer
    dst_ptr,  # Destination buffer pointer
    size: tl.constexpr,  # Data size
    tile_size: tl.constexpr,  # Tile size
    reduction_op: tl.constexpr,  # 'sum', 'max', 'min'
    CLUSTER_SIZE: tl.constexpr,  # Number of blocks
    BLOCK_SIZE: tl.constexpr,  # Threads per block
):
    """
    Cluster-Reduce:  Gather and reduce data across cluster blocks

    Combines gather and reduction operations. Each block gathers data
    from neighbors and performs element-wise reduction (sum/max/min).

    Args:
        src_ptr:  Source data in shared memory
        dst_ptr:  Destination buffer in shared memory
        size: Number of elements to process
        tile_size: Processing tile size
        reduction_op:  Type of reduction ('sum', 'max', 'min')
        CLUSTER_SIZE:  Cluster size
        BLOCK_SIZE: Block size
    """
    cluster = ClusterGroup()
    cluster_block_id = cluster.block_rank()
    tid = tl.program_id(axis=1)

    # Iterate through cluster blocks
    for i in range(1, CLUSTER_SIZE - 1):
        # Calculate neighbor block
        dst_block = (cluster_block_id + i) % CLUSTER_SIZE
        neighbor_offset = dst_block * tile_size

        # === GATHER PHASE ===
        if tid == 0:
            offsets = tl.arange(0, size)
            src_data = tl.load(src_ptr + offsets)
            tl.store(dst_ptr + neighbor_offset + offsets, src_data)

        cluster.sync()

        # === REDUCE PHASE ===
        # Each thread processes elements
        num_elements = tile_size // 2
        if tid < num_elements:
            # Load source and destination
            offset = tid * 2
            src_vec = tl.load(src_ptr + offset, mask=offset < size, other=0.0)
            dst_vec = tl.load(dst_ptr + offset, mask=offset < size, other=0.0)

            # Perform reduction
            if reduction_op == "sum":
                result = src_vec + dst_vec
            elif reduction_op == "max":
                result = tl.maximum(src_vec, dst_vec)
            elif reduction_op == "min":
                result = tl.minimum(src_vec, dst_vec)

            # Store result back
            tl.store(src_ptr + offset, result, mask=offset < size)

        cluster.sync()


# ===== Cluster-Reduce for Scalars (Attention softmax use case) =====
@triton.jit
def cluster_reduce_scalar(
    value,  # Local scalar value
    shared_mem_ptr,  # Pointer to shared memory for cluster exchange
    reduction_op: tl.constexpr,  # 'sum', 'max'
    CLUSTER_SIZE: tl.constexpr,
):
    """
    Cluster-Reduce for scalar values (e.g., max/sum in attention)

    Used for reducing scalar statistics like max logits or sum of
    exponentials in softmax computation across cluster blocks.

    Args:
        value: Local scalar value to reduce
        shared_mem_ptr: Shared memory for inter-block communication
        reduction_op: 'sum' or 'max'
        CLUSTER_SIZE: Number of blocks in cluster

    Returns:
        Reduced value across all cluster blocks
    """
    cluster = ClusterGroup()
    cluster_block_id = cluster.block_rank()
    tid = tl.program_id(axis=1)

    # Store local value to shared memory
    if tid == 0:
        tl.store(shared_mem_ptr + cluster_block_id, value)

    cluster.sync()

    # Iterate and accumulate from neighbors
    for i in range(1, CLUSTER_SIZE - 1):
        dst_block = (cluster_block_id + i) % CLUSTER_SIZE

        if tid == 0:
            # Map to neighbor's shared memory
            neighbor_value = tl.load(shared_mem_ptr + dst_block)
            local_value = tl.load(shared_mem_ptr + cluster_block_id)

            # Perform atomic reduction
            if reduction_op == "sum":
                result = local_value + neighbor_value
            elif reduction_op == "max":
                result = tl.maximum(local_value, neighbor_value)

            tl.store(shared_mem_ptr + cluster_block_id, result)

        cluster.sync()

    # Return reduced value
    return tl.load(shared_mem_ptr + cluster_block_id)


# ===== High-Level Kernel Example:  Attention with Cluster-Reduce =====
@triton.jit
def fused_attention_cluster(
    Queries,
    Keys,
    Values,
    Output,
    stride_qm,
    stride_qk,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    stride_om,
    stride_ok,
    M,
    N,
    K,
    CLUSTER_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused attention kernel using cluster-level primitives

    Demonstrates how cluster_reduce is used in attention computation
    to aggregate statistics (max, sum) across cluster blocks.
    """
    # Get cluster and block info
    cluster = ClusterGroup()
    cluster_block_id = cluster.block_rank()
    pid_m = tl.program_id(0)

    # Allocate shared memory for cluster communication
    cluster_max = tl.zeros([1], dtype=tl.float32)
    cluster_sum = tl.zeros([1], dtype=tl.float32)

    # Compute Q @ K^T
    qk_offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    qk_offset_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    local_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    local_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Compute attention scores
    for n in range(0, N, BLOCK_N):
        # Load Q, K
        q = tl.load(
            Queries
            + qk_offset_m[:, None] * stride_qm
            + qk_offset_k[None, :] * stride_qk
        )
        k = tl.load(Keys + n * stride_kn + qk_offset_k[:, None] * stride_kk)

        # QK^T
        qk = tl.dot(q, k)

        # Track max for numerical stability
        row_max = tl.max(qk, axis=1)
        local_max = tl.maximum(local_max, row_max)

    # === CLUSTER-REDUCE:  Max across blocks ===
    global_max = cluster_reduce_scalar(local_max, cluster_max, "max", CLUSTER_SIZE)

    # Compute exp(qk - max) and sum
    for n in range(0, N, BLOCK_N):
        qk = tl.load(...)  # Re-compute or load from cache
        qk_scaled = tl.exp(qk - global_max[:, None])
        local_sum += tl.sum(qk_scaled, axis=1)
        acc += tl.dot(qk_scaled, Values)

    # === CLUSTER-REDUCE: Sum across blocks ===
    global_sum = cluster_reduce_scalar(local_sum, cluster_sum, "sum", CLUSTER_SIZE)

    # Normalize and write output
    acc = acc / global_sum[:, None]
    tl.store(Output + qk_offset_m[:, None] * stride_om, acc)


# ===== Python API for launching cluster kernels =====
def launch_cluster_reduce(src_tensor, dst_tensor, cluster_size=4, reduction_op="sum"):
    """
    Python API to launch cluster reduce kernel

    Args:
        src_tensor: Source PyTorch tensor
        dst_tensor: Destination PyTorch tensor
        cluster_size: Number of blocks per cluster
        reduction_op: 'sum', 'max', or 'min'
    """
    assert src_tensor.is_cuda

    M, N = src_tensor.shape
    BLOCK_SIZE = 256

    def grid(meta):
        return (
            triton.cdiv(M, cluster_size),
            BLOCK_SIZE,
        )

    cluster_reduce[grid](
        src_tensor,
        dst_tensor,
        size=N,
        tile_size=N // cluster_size,
        reduction_op=reduction_op,
        CLUSTER_SIZE=cluster_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# ===== Usage Example =====
if __name__ == "__main__":
    import torch

    # Create sample tensors
    device = torch.device("cuda")
    M, N = 1024, 4096
    src = torch.randn(M, N, device=device, dtype=torch.float16)
    dst = torch.zeros(M, N, device=device, dtype=torch.float16)

    # Launch cluster reduce
    launch_cluster_reduce(src, dst, cluster_size=4, reduction_op="sum")

    print("Cluster reduce completed!")
    print(f"Result shape: {dst.shape}")
