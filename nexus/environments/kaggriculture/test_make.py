print("1. Script started")

from kaggle_environments import make
from nexus.environments.kaggriculture.environment import KaggricultureEnvironment

print("2. Kaggle import successful")

print("3. About to create environment")

env = make(
    "kaggriculture",
    configuration={
        "episodeSteps": 720
    },
    debug=True,
)

print("4. Environment created successfully")
print(env)
def test_environment_accepts_seed():
    env = KaggricultureEnvironment(
        episode_steps=24,
        debug=False,
        seed=123,
    )

    assert env.env.info["seed"] == 123

def test_environment_same_seed_is_reproducible():
    env_a = KaggricultureEnvironment(
        episode_steps=24,
        debug=False,
        seed=123,
    )
    env_b = KaggricultureEnvironment(
        episode_steps=24,
        debug=False,
        seed=123,
    )

    assert env_a.env.info["seed"] == env_b.env.info["seed"] == 123

def test_environment_same_seed_same_actions_same_result():
    env_a = KaggricultureEnvironment(
        episode_steps=24,
        debug=False,
        seed=123,
    )
    env_b = KaggricultureEnvironment(
        episode_steps=24,
        debug=False,
        seed=123,
    )

    env_a.reset()
    env_b.reset()

    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [],
    }

    for _ in range(24):
        if env_a.state[0].status != "ACTIVE":
            break
        env_a.step([action, action])
        env_b.step([action, action])

    observation_a = env_a.state[0].observation
    observation_b = env_b.state[0].observation

    assert observation_a == observation_b
