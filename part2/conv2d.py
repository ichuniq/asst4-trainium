import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.
Returns:
    X_out: Output tensor [batch_size, out_channels, out_pool_height, out_pool_width]

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

** assume the size of the weights would always be such that it can completely fit inside SBUF.
** assume (input_height - filter_height + 1) % pool_size == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

Map Convolution to a series of indenpendent Matmul
- Reshape input X to: [in_channels, input_heigt*input_width], then multiplied it by each position of the filters,
  where i and j respectively range from 0 to filter_height-1 and from 0 to filter_width-1.
- Each filter slice has shape [out_channels, in_channels], and the resulting matrix multiplication contracts 
  along the Input Channels dimension.
- To align the input with each filter slice, the input must be shifted by an offset corresponding to the filter’s 
  current position (i, j).
- Put filter slice at LHS (stationary) and correspond shifted_X to RHS (moving)
- Shape: [out_channels, in_channels] @ [in_channels, input_heigt*input_width] = [out_channels, input_heigt*input_width]
- Accumulate this matmul result

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size

    N = out_height * out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Reshape input X and W
    # X = X.reshape((batch_size, in_channels, input_height * input_width))
    # W = W.reshape((out_channels, in_channels, filter_height * filter_width))

    # input_flat = X.shape[2]
    # assert input_flat == input_height * input_width

    # Various tiling dimensions (You may want to define more of them)
    TILE_M = 128  # out_channels tile size
    TILE_K = nl.tile_size.pmax  # 128, in_channels tile size
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Calculate number of tiles
    out_ch_n_tiles = out_channels // TILE_M
    in_ch_n_tiles = in_channels // TILE_K

    # Define indices for matmul result
    ip_oc = nl.arange(TILE_M)[:, None]  # Output channels index (partition)
    if_out = nl.arange(N)[None, :]  # Output pixels index (free)

    # Process the images in batches
    for b in nl.affine_range(batch_size):

        # For each out_channel tile
        for m in nl.affine_range(out_ch_n_tiles):
            # Initialize result accumulator for this output channel tile
            res_psum = nl.zeros(
                (par_dim(TILE_M), N),
                dtype=np.float32, 
                buffer=nl.psum
            )

            # Load weights for this output channel tile
            W_sbuf_tile = nl.load(W[m * TILE_M:(m + 1) * TILE_M])

            # For each input channel tile
            for k in nl.affine_range(in_ch_n_tiles):
                # Prepare input patches for all filter positions
                X_batch_tile = nl.load(
                    X[b, k * TILE_K:(k + 1) * TILE_K, 
                      :input_height,
                      :input_width]
                )

                # Loop over each filter position
                for i in nl.affine_range(filter_height):
                    for j in nl.affine_range(filter_width):
                        # Get filter slice for current (i,j), shape: [out_channels, in_channels]
                        W_slice = W_sbuf_tile[:, k * TILE_K:(k + 1) * TILE_K, i, j]
                        
                        # Get the corresponding X
                        X_patches = nl.ndarray((par_dim(TILE_K), N), dtype=X.dtype, buffer=nl.sbuf)

                        # Extract the value for this patch row by row
                        for oh in nl.affine_range(out_height):
                            row_start = oh * out_width
                            row_end = (oh + 1) * out_width
                            X_patches[:, row_start:row_end] = X_batch_tile[:, oh + i, j:j + out_width]

                        # Perform matmul
                        res_psum[ip_oc, if_out] += nl.matmul(W_slice, X_patches)
            
            # Copy result to sbuf, [out_ch_tile_size, out_height * out_width]
            res_sbuf = nl.copy(res_psum)

            # Load bias for this output channel tile
            bias_tile = nl.ndarray((par_dim(TILE_M), 1), dtype=bias.dtype, buffer=nl.sbuf)
            bias_tile[:, 0] = nl.load(bias[m * TILE_M:(m + 1) * TILE_M])

            res_sbuf = nl.add(res_sbuf, bias_tile)
            
            # Reshape to out_sbuf (since can't use .reshape in sbuf)
            out_sbuf = nl.ndarray((TILE_M, out_height, out_width), dtype=X.dtype, buffer=nl.sbuf)
            for oc in nl.affine_range(TILE_M):
                for oh in nl.affine_range(out_height):
                    row_start = oh * out_width
                    for ow in nl.affine_range(out_width):
                        out_sbuf[oc, oh, ow] = res_sbuf[oc, row_start + ow]
            
            nl.store(X_out[b, m * TILE_M:(m + 1) * TILE_M], value=out_sbuf)

    return X_out

