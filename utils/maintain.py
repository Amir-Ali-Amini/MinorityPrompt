def clean_gpu():
    import torch
    import gc

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Run garbage collector
    gc.collect()

    # Check memory
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
