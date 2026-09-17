def accumulation_window_size(batch_index, num_batches, accumulation_steps):
    """Return the actual number of micro-batches in the current update window."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")
    if not 0 <= batch_index < num_batches:
        raise IndexError("batch_index must refer to an existing batch")
    window_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, num_batches - window_start)


def is_accumulation_boundary(batch_index, num_batches, accumulation_steps):
    """Return whether the current micro-batch completes an update window."""
    accumulation_window_size(batch_index, num_batches, accumulation_steps)
    return (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == num_batches
