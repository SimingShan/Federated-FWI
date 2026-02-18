import torch
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import Context, NDArrays, Scalar
from src.diffusion_models.loader import build_diffusion_model
from src.regularization.base import RegularizationMethod
from src.core.losses import LossCalculator
from src.federated_learning.flwr_utils import (
    ndarrays_to_tensor, tensor_to_ndarrays,
)
from src.utils.data_trans import set_seed

class FwiClient(fl.client.NumPyClient):
    def __init__(self, cid: str, device: torch.device,
                 fwi_forward,
                 local_data: torch.Tensor,
                 diffusion_state_dict: Optional[dict] = None,
                 diffusion_model_structure_args: Optional[dict] = None):

        self.cid = int(cid)
        self.device = device
        self.fwi_forward = fwi_forward
        self.local_data = local_data.to(self.device)
        self.diffusion_model = None

        if diffusion_state_dict is not None and diffusion_model_structure_args is not None:
            self.diffusion_model = build_diffusion_model(diffusion_model_structure_args, self.device)
            self.diffusion_model.load_state_dict(diffusion_state_dict)
            self.diffusion_model.eval()

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:

        set_seed(int(config.get("seed", 42)) + int(config["server_round"]))
        local_model = ndarrays_to_tensor(parameters, self.device).requires_grad_(True)
        
        local_epochs: int = int(config["local_epochs"])
        local_lr: float = float(config["local_lr"])
        total_rounds: int = int(config["total_rounds"])
        server_round: int = int(config["server_round"])
        regularization = config["regularization"]
        reg_lambda = config["reg_lambda"]
        num_patches = int(config.get("num_patches", 0)) or None
        
        regularization_method = RegularizationMethod(regularization, self.diffusion_model, num_patches=num_patches)
        loss_calc = LossCalculator(regularization_method)
        global_step = (server_round - 1) * local_epochs
        optimizer_local = torch.optim.Adam([local_model], lr=local_lr)
        for pg in optimizer_local.param_groups:
            pg['initial_lr'] = local_lr
        scheduler_local = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_local,
            T_max=local_epochs * total_rounds,
            eta_min=0.0,
            last_epoch=global_step - 1,
        )

        for _ in range(local_epochs):

            optimizer_local.zero_grad()
            model_input = local_model[:, :, 1:-1, 1:-1]
            predicted_seismic = self.fwi_forward(model_input)

            obs_loss = loss_calc.observation_loss(predicted_seismic, self.local_data)
            raw_reg_loss = loss_calc.regularization_loss(local_model)
            total_loss = loss_calc.total_loss(obs_loss, raw_reg_loss, reg_lambda)

            total_loss.sum().backward()
            optimizer_local.step()
            scheduler_local.step() 
            local_model.data.clamp_(-1, 1)

        updated_ndarrays = tensor_to_ndarrays(local_model)
        num_examples = self.local_data.numel()
        
        return updated_ndarrays, num_examples, {
            "obs_loss": float(obs_loss.mean().item()),
            "reg_loss": float(raw_reg_loss.mean().item()),
            "total_loss": float(total_loss.mean().item()),
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        return 0.0, 0, {}
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return []

def client_fn_factory(partitions: List[torch.Tensor],
                      device, fwi_forward,
                      diffusion_state_dict: Optional[dict] = None,
                      diffusion_args: Optional[dict] = None,
                      per_client_fwi_forwards: Optional[List] = None,
                      ):

    def client_fn(context: Context) -> fl.client.Client:

        client_id = int(context.node_config["partition-id"])
        local_data_for_client = partitions[client_id]

        # Use per-client forward solver if available, else shared
        fwi = per_client_fwi_forwards[client_id] if per_client_fwi_forwards else fwi_forward

        fwi_client_instance = FwiClient(
            cid=str(client_id),
            device=device,
            fwi_forward=fwi,
            local_data=local_data_for_client,
            diffusion_state_dict=diffusion_state_dict,
            diffusion_model_structure_args=diffusion_args,
        )

        return fwi_client_instance.to_client()

    return client_fn
