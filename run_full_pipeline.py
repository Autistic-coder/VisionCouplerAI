import subprocess
import sys


def run_step(command: list[str], label: str) -> bool:
    print(f"\n=== {label} ===")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"{label} failed. Stopping pipeline.")
        return False
    return True


def main() -> None:
    python = sys.executable
    if not run_step([python, "organize_annotated_dataset.py"], "Organize annotated dataset"):
        return
    if not run_step([python, "validate_dataset.py"], "Validate dataset"):
        return
    if not run_step([python, "train_improved_yolo.py"], "Train YOLOv8 model"):
        return

    print("\nFull pipeline completed.")
    print("Next step: python desktop_app.py")


if __name__ == "__main__":
    main()
