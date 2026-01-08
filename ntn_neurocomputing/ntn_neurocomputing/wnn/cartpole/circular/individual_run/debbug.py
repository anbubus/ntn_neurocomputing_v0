import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog
import numpy as np
import random
import os

# from seu_modulo import NTNModel  # ajuste o caminho para o seu modelo real
from ntn_neurocomputing.wnn.ntn_model import NTNModel  # exemplo

if __name__ == "__main__":
    ray.init(local_mode=True, ignore_reinit_error=True)

    # Registrar o modelo customizado
    ModelCatalog.register_custom_model("ntn_model", NTNModel)

    # Semente
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)

    # Configuração simplificada
    config = (
        PPOConfig()
        .framework("torch")
        .environment(env="CartPole-v1")
        .env_runners(num_env_runners=1)   # reduzido para debug
        .training(
            lr=0.003,
            num_epochs=1,
            minibatch_size=128,
            model={
                "custom_model": "ntn_model",
                "custom_model_config": {
                    "tuple_size": 8,
                    "encoding": {
                        "enc_type": "circular",
                        "resolution": 64,
                        "min": -1.5,
                        "max": 1.5,
                    }
                },
            },
        )
        # Desativa o novo API stack para suportar ModelV2
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    print("✅ Configuração criada. Inicializando PPO...")

    try:
        algo = PPO(config=config)
        print("✅ PPO inicializado com sucesso.")
        result = algo.train()
        print("🏁 Resultado do primeiro treino:")
        print(result)
    except Exception as e:
        import traceback
        print("❌ ERRO DETECTADO:")
        traceback.print_exc()
    finally:
        ray.shutdown()



