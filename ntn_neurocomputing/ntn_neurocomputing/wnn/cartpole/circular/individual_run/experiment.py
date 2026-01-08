import os
import random
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog


# import os

# import numpy as np
# import random

from ntn_neurocomputing.wnn.ntn_model import NTNModel

# import ray
# from ray import tune
# from ray.rllib.models import ModelCatalog

# importa/define NTNModel antes
# from seu_modulo import NTNModel

if __name__ == "__main__":
    # inicia o ray (pode usar local_mode=True se quiser)
    ray.init()

    # registre o seu modelo antigo (ModelV2)
    ModelCatalog.register_custom_model("ntn_model", NTNModel)

    seed = 12345678
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # --- CRIA CONFIG COM PPOConfig E DESATIVA O NOVO API STACK ---
    config = (
        PPOConfig()
        .framework("torch")
        .environment(env="CartPole-v1")
        .training(
            lr=0.003,
            num_sgd_iter=1,
            minibatch_size=128,
        )
        .env_runners(num_env_runners=2)  # num_workers equivalent
        # desativa novo API stack (compatibilidade com custom_model)
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        # configura o modelo (ModelV2 style)
        .training(model={
            "custom_model": "ntn_model",
            "custom_model_config": {
                "seed": tune.sample_from(
                    lambda _: int(rng.integers(1_000, int(1e6)))
                ),
                "tuple_size": 8,
                "encoding": {
                    "enc_type": "circular",
                    "resolution": 64,
                    "min": -1.5,
                    "max": 1.5
                }
            }
        })
    )

    # Transforme para dict para passar ao tune.run
    config_dict = config.to_dict()

    # Evite resumir experimentos antigos enquanto testa:
    analysis = tune.run(
        PPO,  # use a classe PPO aqui
        name="experiment",
        storage_path=os.path.abspath(os.path.dirname(__file__)),
        resume=False,            # <- importantíssimo para testar a mudança
        num_samples=20,
        config=config_dict,
        stop={'timesteps_total': 1_000_000},
        checkpoint_freq=5,
        checkpoint_at_end=True
    )

    ray.shutdown()












# import os

# import numpy as np
# import random

# from ntn_neurocomputing.wnn.ntn_model import NTNModel

# import ray
# from ray import tune
# from ray.rllib.models import ModelCatalog



# if __name__ == "__main__":
#     ray.init()

#     ModelCatalog.register_custom_model("ntn_model", NTNModel)

#     seed = 12345678

#     random.seed(seed)
#     np.random.seed(seed)
#     rng = np.random.default_rng(seed)

#     analysis = tune.run(
#         "PPO",
#         name="experiment",
#         storage_path=os.path.abspath(os.path.dirname(__file__)),
#         resume="AUTO",
#         num_samples=20,
#         config={
#             "env": "CartPole-v1",
#             "framework": "torch",
#             "num_workers": 2,
#             "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
#             "lr": 0.003,
#             "observation_filter": "MeanStdFilter",
#             "num_sgd_iter": 1,
#             "sgd_minibatch_size": 128,

#             # <<< ADIÇÃO IMPORTANTE: desativa o novo API stack para compatibilidade
#             "api_stack": {
#                 "enable_rl_module_and_learner": False,
#                 "enable_env_runner_and_connector_v2": False
#             },

#             "model": {
#                 "custom_model": "ntn_model",
#                 "custom_model_config": {
#                     "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
#                     "tuple_size": 8,
#                     "encoding": {
#                         "enc_type": "circular",
#                         "resolution": 64,
#                         "min": -1.5,
#                         "max": 1.5
#                     }
#                 },
#             },
#         },
#         stop={'timesteps_total': 1_000_000},
#         checkpoint_freq=5,
#         checkpoint_at_end=True
#     )

#     ray.shutdown()

    # analysis = tune.run(
    #     "PPO",
    #     name="experiment",
    #     storage_path=os.path.abspath(os.path.dirname(__file__)),
    #     resume="AUTO",
    #     num_samples=20,
    #     config={
    #         "env": "CartPole-v1",
    #         "framework": "torch",
    #         "num_workers": 2,
    #         "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
    #         "lr": 0.003,
    #         "observation_filter": "MeanStdFilter",
    #         "num_sgd_iter": 1,
    #         "sgd_minibatch_size": 128,
    #         "model": {
    #             "custom_model": "ntn_model",
    #             "custom_model_config": {
    #                 "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
    #                 "tuple_size": 8,
    #                 "encoding": {
    #                     "enc_type": "circular",
    #                     "resolution": 64,
    #                     "min": -1.5,
    #                     "max": 1.5
    #                 }
    #             },
    #         },
    #     },
    #     stop={ 'timesteps_total': 1_000_000 },
    #     checkpoint_freq=5,
    #     checkpoint_at_end=True
    # )

    # ray.shutdown()