import warnings
import torch

def generate_stats(output_token_ids, start_time, end_time):
    total_time = end_time - start_time
    print(f"\n\nTime: {total_time: .2f} sec")
    print(f"{int(output_token_ids.numel() / total_time)} token/sec")


    for name, backend in (("CUDA", getattr(torch, "cuda")), ("XPU", getattr(torch, "xpu", None))):
        if backend is not None and backend.is_available():

            ## check wheter we are currently using this backend
            device_type = output_token_ids.device.type

            if device_type != name.lower():
                warnings.warn(
                    f"{name} is not available, but the tensors are on"
                    f"{device_type}. Memory stats may be 0"
                )

            ## synchronize
            if hasattr(backend, "synchronize"):
                backend.synchronize()

            max_memory_bytes = backend.max_memory_allocated()
            max_memory_gb = max_memory_bytes / (1024 ** 3)
            print(f"Max {name} memory allocated: {max_memory_gb:.2f} GB")
            backend.reset_peak_memory_stats()
