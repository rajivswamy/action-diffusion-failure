from src.envs.sim_pusht import PushTEnv, PushTImageEnv


def make_env(name: str, args) -> PushTEnv:
    """
    Factory function to create an instance of PushTEnv or PushTImageEnv.

    :param name: Name of the environment to create.
    :param kwargs: Additional keyword arguments for environment configuration.
    :return: An instance of PushTEnv or PushTImageEnv.
    """

    if name == "PushTEnv":
        from src.envs.sim_pusht import PushTEnv

        return PushTEnv.make_env(args)
    elif name == "PushTImageEnv":
        from src.envs.sim_pusht import PushTImageEnv
        
        return PushTImageEnv.make_env(args)
    else:
        raise ValueError(f"Unknown environment name: {name}")