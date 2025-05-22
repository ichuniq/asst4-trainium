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
- To align the input with each filter slice, the input must be shifted by an offset corresponding to the filter's
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

    # Various tiling dimensions
    TILE_M = 128  # out_channels tile size
    TILE_K = nl.tile_size.pmax  # 128, in_channels tile size
    
    # Tiling for spatial (out height and width) dimension
    TILE_H = min(out_height, nl.tile_size.gemm_moving_fmax // out_width) # num rows per tile
    TILE_HW = TILE_H * out_width # total pixels per spatial tile
    
    # Calculate number of tiles
    out_ch_n_tiles = out_channels // TILE_M
    in_ch_n_tiles = in_channels // TILE_K
    hw_n_tiles = (out_height + TILE_H - 1) // TILE_H  # ceiling division

    # Define indices for tiled matmul result
    ip_oc = nl.arange(TILE_M)[:, None]  # Output channels index (partition)
    if_out = nl.arange(TILE_HW)[None, :]  # Output pixels index (free)

    # Preload bias to SBUF
    bias_tiles = nl.ndarray((TILE_M, out_ch_n_tiles), dtype=bias.dtype, buffer=nl.sbuf)
    for m in nl.affine_range(out_ch_n_tiles):
        bias_tiles[:, m] = nl.load(bias[m * TILE_M:(m + 1) * TILE_M])

    # Preload W to SBUF
    W_tiles = nl.ndarray((TILE_M, out_ch_n_tiles, in_channels, filter_height, filter_width), dtype=bias.dtype, buffer=nl.sbuf)
    for m in nl.affine_range(out_ch_n_tiles):
        for ic in nl.affine_range(in_channels):
            W_tiles[:, m, ic, :, :] = nl.load(W[m * TILE_M:(m + 1) * TILE_M, ic, :, :])


    # Process the images in batches
    for b in nl.affine_range(batch_size):

        # Process spatial tiles first
        for hw in nl.affine_range(hw_n_tiles):
            # Starting position for this spatial tile
            start_h = hw * TILE_H
            
            # Fixed input height for this tile including filter overlap
            input_tile_height = TILE_H + filter_height - 1
            
            # Preload all input channel tiles for this spatial tile
            X_batch_tile = nl.ndarray((in_ch_n_tiles, par_dim(TILE_K), input_tile_height, input_width),
                                  dtype=X.dtype, buffer=nl.sbuf)
            for k in nl.affine_range(in_ch_n_tiles):
                # Use masking to handle boundary conditions
                i_par, i_row, i_col = nl.mgrid[0:TILE_K, 0:input_tile_height, 0:input_width]
                mask = (start_h + i_row) < input_height
                X_batch_tile[k] = nl.load(
                    X[b, k * TILE_K + i_par, start_h + i_row, i_col],
                    mask=mask
                )
            
            # Process output channel tiles
            for m in nl.affine_range(out_ch_n_tiles):
                # Get bias for this output channel tile
                bias_tile = nl.ndarray((TILE_M, 1), dtype=bias.dtype, buffer=nl.sbuf)
                bias_tile[:, 0] = nl.copy(bias_tiles[:, m])

                # Load weights for this output channel tile
                W_tile = nl.copy(W_tiles[:, m, :, :])
                
                # Initialize result accumulator for this spatial tile
                res_psum = nl.zeros((TILE_M, TILE_HW), dtype=nl.float32, buffer=nl.psum)
                
                # Loop over each filter position
                for i in nl.affine_range(filter_height):
                    for j in nl.affine_range(filter_width):
                        # Process all input channel tiles for this filter position
                        for k in nl.affine_range(in_ch_n_tiles):
                            # Get filter slice for current (i,j), shape: [out_channels, in_channels]
                            W_slice = W_tile[:, k * TILE_K:(k + 1) * TILE_K, i, j]
                            
                            # Get the corresponding X
                            X_patches = nl.ndarray((par_dim(TILE_K), TILE_HW), dtype=X.dtype, buffer=nl.sbuf)

                            # Extract the value for this patch row by row
                            for h in nl.affine_range(TILE_H):
                                if start_h + h >= out_height: break
                                    
                                row_start = h * out_width
                                row_end = (h + 1) * out_width
                                X_patches[:, row_start:row_end] = X_batch_tile[k, :, h + i, j:j + out_width]

                            # Perform matmul
                            res_psum[ip_oc, if_out] += nl.matmul(W_slice, X_patches)
                
                # Add bias
                res_psum = nl.add(res_psum, bias_tile)
                
                # Copy result to out_sbuf, size: [out_ch_tile_size, TILE_H, out_width]
                out_sbuf = nl.copy(res_psum.reshape((TILE_M, TILE_H, out_width)))
                
                # Store result for this spatial tile, ensuring don't write beyond output bounds
                i_par, i_h, i_w = nl.mgrid[0:TILE_M, 0:TILE_H, 0:out_width]
                mask = (start_h + i_h) < out_height
                nl.store(
                    X_out[b, m * TILE_M + i_par, start_h + i_h, i_w],
                    value=out_sbuf,
                    mask=mask
                )

    return X_out

