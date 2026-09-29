from kaggle_environments import make


class KaggricultureEnvironment:
    """NEXUS adapter for the Kaggriculture simulation."""

    def __init__(self, episode_steps=720, debug=True, seed=None):
        self.env = make(
            "kaggriculture",
            configuration={"episodeSteps": episode_steps, "seed": seed},
            debug=debug,
        )

    def reset(self):
        """Start a new Kaggriculture episode."""
        return self.env.reset()

    def step(self, actions):
        """Advance the simulation by one step."""
        return self.env.step(actions)

    @property
    def state(self):
        """Return the current Kaggriculture environment state."""
        return self.env.state

    @property
    def configuration(self):
        """Return the environment configuration."""
        return self.env.configuration

    @property
    def specification(self):
        """Return the environment specification."""
        return self.env.specification
