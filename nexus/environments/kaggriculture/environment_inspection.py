print("NEXUS INSPECTION SCRIPT STARTED")

from kaggle_environments import make


def main():
    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": 720
        },
        debug=True,
    )

    print("=" * 70)
    print("NEXUS — Kaggriculture Environment")
    print("=" * 70)

    print("\nEnvironment:")
    print(env)

    print("\nConfiguration:")
    print(env.configuration)

    print("\nSpecification:")
    print(env.specification)


if __name__ == "__main__":
    main()