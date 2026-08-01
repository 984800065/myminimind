import random

import swanlab


def run_swanlab_smoke() -> None:
    """Manual integration smoke; never contact SwanLab during pytest collection."""
    run = swanlab.init(
        project="my-project",
        config={
            "learning_rate": 0.01,
            "epochs": 10,
        },
    )

    print(f"学习率为{run.config.learning_rate}")
    offset = random.random() / 5

    for epoch in range(2, run.config.epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
        swanlab.log({"accuracy": acc, "loss": loss})


if __name__ == "__main__":
    run_swanlab_smoke()
