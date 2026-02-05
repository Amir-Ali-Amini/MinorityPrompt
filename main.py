def main():
    print("Hello from minorityprompt!")


from minority_gen.grid_search import random_search
from minority_gen.task import sharif_task

if __name__ == "__main__":
    main()
    sharif_task(model="sd15", use_lightning=True)
    # random_search(
    #     models=["sdxl_lightning"],
    #     n_configs=25,
    #     n_samples=1,
    #     seed=42,
    # )
