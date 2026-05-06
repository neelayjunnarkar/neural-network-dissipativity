import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override

import lti_controllers
from models.RINN import RINN
from theta_dissipativity import construct_closed_loop, construct_dissipativity_matrix
from utils import from_numpy
from variable_structs import ControllerThetaParameters


class SoftDissipativeRINN(RINN):
    """
    A RINN controller with soft dissipativity enforcement via a penalty term in the RL loss.

    Instead of projecting onto the dissipative set after each gradient step (as in
    DissipativeSimplestRINN), this model adds:

        soft_weight * relu(lambda_max(F(theta, P, Lambda)))

    to the PPO loss, where F is the dissipativity matrix from theta_dissipativity.
    A well-posedness penalty is also included.

    P and Lambda can each be learnable (default) or fixed at LTI initialization values,
    controlled by free_P and free_Lambda in model_config.

    Required model_config keys:
        soft_weight: float    -- penalty coefficient
        plant: env class      -- environment class (same as DissipativeSimplestRINN)
        plant_config: dict    -- environment config

    Optional model_config keys:
        free_P: bool          -- if True (default), P is a learnable parameter
        free_Lambda: bool     -- if True (default), Lambda is a learnable parameter
        eps: float            -- numerical conditioning for P (default 1e-6)
        lti_initializer: str  -- key into lti_controllers.controller_map (optional)
        lti_initializer_kwargs: dict
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs
    ):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, *args, **kwargs
        )

        cfg = model_config["custom_model_config"]

        assert "soft_weight" in cfg, "soft_weight is required in custom_model_config"
        self.soft_weight = cfg["soft_weight"]
        self.free_P = cfg.get("free_P", True)
        self.free_Lambda = cfg.get("free_Lambda", True)
        self._eps = cfg.get("eps", 1e-6)

        assert "plant" in cfg and "plant_config" in cfg, (
            "plant and plant_config are required"
        )
        plant = cfg["plant"](cfg["plant_config"])
        np_plant_params = plant.get_params()
        device = self.log_stds.device

        self.plant_params = np_plant_params.np_to_torch(device=device)

        # Precompute LDeltap = sqrt(MDeltapvv) factor (for IQC on plant nonlinearity)
        Dm, Vm = np.linalg.eigh(np_plant_params.MDeltapvv)
        LDeltap_np = np.diag(np.sqrt(np.maximum(Dm, 0.0))) @ Vm.T
        self.register_buffer("LDeltap", from_numpy(LDeltap_np, device=device))

        # Precompute LX = sqrt(-Xee) factor (for supply rate)
        Dx, Vx = np.linalg.eigh(-np_plant_params.Xee)
        LX_np = np.diag(np.sqrt(np.maximum(Dx, 0.0))) @ Vx.T
        self.register_buffer("LX", from_numpy(LX_np, device=device))

        # Supply rate matrices as buffers
        self.register_buffer("Xdd", from_numpy(np_plant_params.Xdd, device=device))
        self.register_buffer("Xde", from_numpy(np_plant_params.Xde, device=device))

        P_size = np_plant_params.Ap.shape[0] + self.state_size
        P_init = np.eye(P_size, dtype=np.float32)

        # Optional LTI initialization (same pattern as DissipativeSimplestRINN)
        lti_initializer = cfg.get("lti_initializer", None)
        if lti_initializer is not None:
            print(
                f"SoftDissipativeRINN: initializing from {lti_initializer} LTI controller."
            )
            lti_kwargs = dict(cfg.get("lti_initializer_kwargs", {}))
            lti_kwargs["state_size"] = self.state_size
            lti_kwargs["input_size"] = self.input_size
            lti_kwargs["output_size"] = self.output_size
            lti_controller, info = lti_controllers.controller_map[lti_initializer](
                np_plant_params, **lti_kwargs
            )
            lti_controller = lti_controller.np_to_torch(device=device)
            # Override RINN parameters with LTI values; zero out nonlinear paths
            self.A_T = nn.Parameter(lti_controller.Ak.t())
            self.By_T = nn.Parameter(lti_controller.Bky.t())
            self.Cu_T = nn.Parameter(lti_controller.Cku.t())
            self.Duy_T = nn.Parameter(lti_controller.Dkuy.t())
            self.Bw_T = nn.Parameter(
                torch.zeros(self.nonlin_size, self.state_size, device=device)
            )
            self.Cv_T = nn.Parameter(
                torch.zeros(self.state_size, self.nonlin_size, device=device)
            )
            self.Dvw_T = nn.Parameter(
                torch.zeros(self.nonlin_size, self.nonlin_size, device=device)
            )
            self.Dvy_T = nn.Parameter(
                torch.zeros(self.input_size, self.nonlin_size, device=device)
            )
            self.Duw_T = nn.Parameter(
                torch.zeros(self.nonlin_size, self.output_size, device=device)
            )
            if "P" in info:
                P_init = info["P"].astype(np.float32)

        # P parameterization
        if self.free_P:
            P_chol_init = np.linalg.cholesky(
                P_init + 1e-6 * np.eye(P_size, dtype=np.float32)
            )
            self.P_chol = nn.Parameter(from_numpy(P_chol_init, device=device))
        else:
            self.register_buffer("P_fixed", from_numpy(P_init, device=device))

        # Lambda parameterization: initialize to 0.1 * I
        if self.free_Lambda:
            log_lambda_init = np.log(0.1) * np.ones(self.nonlin_size, dtype=np.float32)
            self.log_lambda = nn.Parameter(torch.tensor(log_lambda_init, device=device))
        else:
            self.register_buffer(
                "Lambda_fixed",
                torch.diag(0.1 * torch.ones(self.nonlin_size, device=device)),
            )

    @property
    def P(self):
        if self.free_P:
            L = torch.tril(self.P_chol)
            return L @ L.t() + self._eps * torch.eye(
                L.shape[0], device=L.device, dtype=L.dtype
            )
        else:
            return self.P_fixed

    @property
    def Lambda(self):
        if self.free_Lambda:
            return torch.diag(torch.exp(self.log_lambda))
        else:
            return self.Lambda_fixed

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        penalty = self._dissipativity_penalty()
        if isinstance(policy_loss, list):
            return [pl + penalty for pl in policy_loss]
        return policy_loss + penalty

    def _dissipativity_penalty(self):
        Ak = self.A_T.t()
        Bkw = self.Bw_T.t()
        Bky = self.By_T.t()
        Ckv = self.Cv_T.t()
        Dkvw = self.Dvw_T.t()
        Dkvy = self.Dvy_T.t()
        Cku = self.Cu_T.t()
        Dkuw = self.Duw_T.t()
        Dkuy = self.Duy_T.t()
        P = self.P
        Lambda = self.Lambda

        controller_params = ControllerThetaParameters(
            Ak, Bkw, Bky, Ckv, Dkvw, Dkvy, Cku, Dkuw, Dkuy, Lambda
        )
        A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, LDelta, Mvw, Mww = construct_closed_loop(
            self.plant_params, self.LDeltap, controller_params, "torch"
        )
        mat = construct_dissipativity_matrix(
            A,
            Bw,
            Bd,
            Cv,
            Dvw,
            Dvd,
            Ce,
            Dew,
            Ded,
            P,
            LDelta,
            Mvw,
            Mww,
            self.Xdd,
            self.Xde,
            self.LX,
            "torch",
        )

        dissipativity_violation = F.relu(torch.linalg.eigvalsh(mat).max())

        # Well-posedness: Lambda @ Dkvw + Dkvw.T @ Lambda - 2*Lambda < 0
        wp_mat = Lambda @ Dkvw + Dkvw.t() @ Lambda - 2 * Lambda
        wp_violation = F.relu(torch.linalg.eigvalsh(wp_mat).max())

        return self.soft_weight * (dissipativity_violation + wp_violation)
