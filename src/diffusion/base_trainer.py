import os
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from hydra.utils import instantiate
from datasets import load_from_disk
from tqdm import trange
import json
from tqdm import tqdm
from copy import deepcopy

from src.diffusion.length_sampler import LengthSampler
from src.models.ema import ExponentialMovingAverage
from src.utils.training_utils import get_stat, mse_loss
from src.utils.logging_utils import log_metric
from src.utils.ddp_utils import reduce_tensor, gather_texts
from src.utils.pylogger import RankedLogger
from src.metrics import compute_ddp_metric

class BaseDiffusionTrainer:
    def __init__(self, config: DictConfig, device: torch.device):
        self.config = config
        self.device = device

        # Model setup
        self.config.model.config.embedding_size = self.config.encoder.config.embedding_dim
        self.score_estimator = instantiate(config=self.config.model).to(self.device)

        encoder_partial = instantiate(self.config.encoder)
        self.encoder = encoder_partial(
            device=device,
            main_config=config,
            add_enc_normalizer=True,
        )

        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), self.config.training.ema_rate)

        self.length_sampler = LengthSampler(
            path=self.config.datasets.length_distribution, 
            max_sequence_len=self.config.datasets.max_sequence_len
        )
        
        # Diffusion setup
        self.scheduler = instantiate(self.config.scheduler)
        self.dynamic = instantiate(self.config.dynamic, scheduler=self.scheduler)
        self.solver = instantiate(
            self.config.solver, 
            dynamic=self.dynamic, 
            score_fn=self.calc_score,
        )
        self.logger = RankedLogger(name="trainer", rank_zero_only=False, rank=self.config.ddp.global_rank)

    def _setup_training_utils(self):
        trainable_params = filter(lambda p: p.requires_grad, self.score_estimator.parameters())
        self.optimizer = instantiate(self.config.optimizer, params=trainable_params)
        self.scheduler = instantiate(self.config.lr_scheduler, optimizer=self.optimizer)
        self.step = 0
        self._setup_ddp()

    def _setup_ddp(self):
        if self.config.ddp.enabled:
            self.ddp_score_estimator = torch.nn.parallel.DistributedDataParallel(
                self.score_estimator,
                device_ids=[dist.get_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

    def _setup_train_data_generator(self):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = load_from_disk(os.path.join(self.config.datasets.data_dir, "train"))

        if self.config.ddp.enabled:
            self.sampler_train = torch.utils.data.DistributedSampler(
                self.train_dataset,
                shuffle=True,
            )
            self.sampler_train.set_epoch(self.step)
        else:
            self.sampler_train = None
        
        self.train_loader = instantiate(self.config.dataloader, dataset=self.train_dataset, sampler=self.sampler_train)

    def _setup_valid_data_generator(self):
        if not hasattr(self, "valid_dataset"):
            self.valid_dataset = load_from_disk(os.path.join(self.config.datasets.data_dir, "test"))

        if self.config.ddp.enabled:
            self.sampler_valid = torch.utils.data.DistributedSampler(
                self.valid_dataset,
                shuffle=False,
            )
        else:
            self.sampler_valid = None   
            
        self.valid_loader = instantiate(self.config.dataloader, dataset=self.valid_dataset, sampler=self.sampler_valid)

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.cuda.FloatTensor(batch_size).uniform_() * (self.dynamic.T - eps) + eps

    def optimizer_step(self, total_loss):   
        self.optimizer.zero_grad()
        total_loss.backward()

        grad_norm = torch.sqrt(sum([torch.sum(t.grad ** 2) for t in self.score_estimator.parameters() if t.requires_grad]))

        if self.config.training.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.score_estimator.parameters(),
                max_norm=self.config.training.grad_clip_norm
            )

        self.optimizer.step()

        self.ema.update(self.score_estimator.parameters())
        self.scheduler.step_update(self.step)

        stat_dict = {
            "grad_norm": grad_norm,
            "lr": self.optimizer.param_groups[0]['lr'],
        }
        return stat_dict

    def log_data(self, loss_dict, stat_dict = None, optimizer_stat_dict = None, is_train: bool = True):
        if is_train:
            loader_name = "train_loader"
        else:
            loader_name = "valid_loader"
        
        # Losses and accuracies
        for key in loss_dict:
            log_metric(key, loader_name, loss_dict[key], self.step)

        # Statistics
        if stat_dict is not None:
            for key in stat_dict:
                log_metric("Statistics", key, stat_dict[key], self.step)

        # Optimizer stats
        if optimizer_stat_dict is not None:
            for key in optimizer_stat_dict:
                log_metric("Optimizer stats", key, optimizer_stat_dict[key], self.step)

    def calc_score(self, x_t, t, mask, x_0_self_cond):
        params = self.dynamic.marginal_params(t)
        x_0 = self.score_estimator(x_t=x_t, time_t=t, attention_mask=mask, x_0_self_cond=x_0_self_cond)
        eps_theta = (x_t - params["mu"] * x_0) / params["std"]
        score = -eps_theta / params["std"]

        return {
            "score": score,
            "x_0": x_0,
            "eps_theta": eps_theta
        }

    def train(self) -> None:
        self.init_checkpoint()

        self._setup_training_utils()
        is_loaded = self.load_checkpoint()
        if is_loaded:
            self.logger.info("Evaluation of loaded checkpoint")
            self.validate()
            self.training_estimation()

        self.ddp_score_estimator.train()

        if self.config.ddp.global_rank == 0:
            self.train_range = trange(self.step + 1, self.config.training.training_iters + 1)
        else:
            self.train_range = range(self.step + 1, self.config.training.training_iters + 1)
        self.train_loader_iter = iter([])
        if self.config.ddp.global_rank == 0:
            self.log_num_parameters()
            self.logger.info(f"Training started with {self.config.training.training_iters} iterations")

        for step in self.train_range:
            self.step = step

            batch = next(self.train_loader_iter, None)
            if batch is None:
                self._setup_train_data_generator()
                self.train_loader_iter = iter(self.train_loader)
                batch = next(self.train_loader_iter, None)
            
            total_loss, loss_dict, stat_dict = self.calc_loss(batch)
            optimizer_stat_dict = self.optimizer_step(total_loss)

            # Logging
            if self.config.ddp.global_rank == 0:
                self.log_data(loss_dict, stat_dict, optimizer_stat_dict, is_train=True)
                self.train_range.set_description(f"total_loss: {total_loss.item():0.3f}")

            # # Save checkpoint
            if self.step % self.config.training.save_interval == 0 and self.config.ddp.global_rank == 0:
                self.save_checkpoint()

            # Evaluation    
            if self.step % self.config.training.eval_interval == 0:
                self.training_estimation()
                self.validate()
                torch.cuda.empty_cache()
    
    def calc_loss(self, batch):
        # Encoder forward
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            clean_X, attention_mask, _ = self.encoder.batch_encode(batch, max_sequence_len=self.config.datasets.max_sequence_len)
            attention_mask = attention_mask.int()

        # Noizing
        batch_size = clean_X.size(0)
        t = self.sample_time(batch_size)
        marg_forward = self.dynamic.marginal(clean_X, t)
        x_t = marg_forward['x_t']

        # self-cond estimation
        x_0_self_cond = torch.zeros_like(x_t, dtype=x_t.dtype)
        loss_x_0_self_cond = torch.tensor(0.0, device=x_t.device)
        if self.config.model.config.use_self_cond:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                x_0_self_cond = self.ddp_score_estimator(
                    x_t=x_t, 
                    time_t=t,
                    attention_mask=attention_mask,
                    x_0_self_cond=x_0_self_cond,
                )
            loss_x_0_self_cond = mse_loss(clean_X, x_0_self_cond, attention_mask)
            x_0_self_cond = x_0_self_cond.clone().detach()

        # model prediction
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x_0 = self.ddp_score_estimator(
                x_t=x_t, 
                time_t=t,
                attention_mask=attention_mask,
                x_0_self_cond=x_0_self_cond,
            )

        # MSE losses
        loss_x_0 = mse_loss(clean_X, x_0, attention_mask)
        total_loss = loss_x_0 + loss_x_0_self_cond
        loss_dict = {
            'total_loss': total_loss,
            'loss_x_0_self_cond': loss_x_0_self_cond,
            'loss_x_0': loss_x_0,
        }

        # Statistics
        with torch.no_grad():
            clean_X_mean, clean_X_std = get_stat(clean_X, attention_mask)
            x_0_mean, x_0_std = get_stat(x_0, attention_mask)
        stat_dict = {
            "clean_X_mean": clean_X_mean,
            "clean_X_std": clean_X_std,
            "x_0_mean": x_0_mean,
            "x_0_std": x_0_std,
        }
        return total_loss, loss_dict, stat_dict

    @torch.no_grad()
    def validate(self) -> None:
        self._setup_valid_data_generator()
        self.ddp_score_estimator.eval()
        self.switch_to_ema()
        
        total_loss = torch.Tensor([0.0])
        valid_count = torch.Tensor([0.0])

        if self.config.ddp.global_rank == 0:
            valid_loader = tqdm(self.valid_loader, desc="Validation")
        else:
            valid_loader = self.valid_loader
        
        for batch in valid_loader:
            batch_size = len(batch["sequence"])
            batch_loss, _, _ = self.calc_loss(batch)
            valid_count += batch_size
            total_loss += batch_loss.item() * batch_size

        valid_count = reduce_tensor(valid_count.cuda())
        total_loss = reduce_tensor(total_loss.cuda())
        total_loss = total_loss / valid_count
        if self.config.ddp.global_rank == 0:
            self.log_data({"total_loss": total_loss}, is_train=False)

        self.switch_back_from_ema()
        self.ddp_score_estimator.train()
        
    def save_checkpoint(self, last: bool = False) -> None:
        if not os.path.exists(self.config.project.diffusion_checkpoints_folder):
            os.makedirs(self.config.project.diffusion_checkpoints_folder)

        prefix_folder = os.path.join(self.config.project.diffusion_checkpoints_folder, self.config.project.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)

        if last:
            prefix = 'last'
        else:
            prefix = str(self.step)

        save_path = os.path.join(prefix_folder, prefix + ".pth")
        state_dict = {   
            "model": self.score_estimator.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
        }
        torch.save(state_dict, save_path)
        self.logger.info(f"Save model to: {save_path}")

    def load_checkpoint(self):
        prefix_folder = os.path.join(self.config.project.diffusion_checkpoints_folder, self.config.project.checkpoints_prefix)

        if not os.path.exists(prefix_folder):
            return False
        
        checkpoint_names = list(os.listdir(prefix_folder))
        checkpoint_names = [str(t).replace(".pth", "") for t in checkpoint_names]
        checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]

        if not checkpoint_names:
            return False

        name = self.config.project.checkpoint_name if hasattr(self.config.project, "checkpoint_name") else None
        if not name:
            name = max(checkpoint_names)
        checkpoint_name = f"{prefix_folder}/{name}.pth"

        load = torch.load(checkpoint_name, map_location="cpu")

        self.ema.load_state_dict(load["ema"])
        self.ema.cuda()
        self.score_estimator.load_state_dict(load["model"])
        self.optimizer.load_state_dict(load["optimizer"])
        self.scheduler.load_state_dict(load["scheduler"])
        self.step = load["step"]

        if self.config.ddp.enabled:
            self._setup_ddp()
        
        self.logger.info(f"Checkpoint is loaded from {checkpoint_name}")
        return True

    def init_checkpoint(self):
        if not self.config.training.init_se:
            return
        
        state_dict = torch.load(self.config.training.init_se)

        base_config = deepcopy(self.config.model)
        base_model = instantiate(config=base_config).to(self.device)
        base_ema = ExponentialMovingAverage(base_model.parameters(), self.config.training.ema_rate)
        base_ema.load_state_dict(state_dict["ema"])
        base_ema.copy_to(base_model.parameters())
        
        # Find differences between state dicts
        score_dict = self.score_estimator.state_dict()
        base_dict = base_model.state_dict()
        
        diff_keys = []
        for key in score_dict:
            if key not in base_dict:
                diff_keys.append(f"Only in score_estimator: {key}")
            elif score_dict[key].shape != base_dict[key].shape:
                diff_keys.append(f"Shape mismatch for {key}: {score_dict[key].shape} vs {base_dict[key].shape}")
                
        for key in base_dict:
            if key not in score_dict:
                diff_keys.append(f"Only in base_model: {key}")
        
        if diff_keys:
            self.logger.info("State dict differences:")
            for diff in diff_keys:
                self.logger.info(diff)

        self.score_estimator.cpu()
        self.score_estimator.load_state_dict(base_model.state_dict(), strict=False)
        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), self.config.training.ema_rate)
        self.ema.cuda()
        self.score_estimator.cuda()

        if self.config.ddp.enabled:
            self._setup_ddp()
        
        self.logger.info(f"Score Estimator is initialized from {self.config.training.init_se}")

    def restore_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.ema.load_state_dict(checkpoint["ema"])
        self.ema.cuda()
        self.score_estimator.load_state_dict(checkpoint["model"])
        self.switch_to_ema()

        self.logger.info(f"Score Estimator is restored from {checkpoint_path}")

    def switch_to_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    def switch_back_from_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.restore(score_model.parameters())

    def training_estimation(self):
        """
        This function generate samples and calculate metrics.
        """
        self.ddp_score_estimator.eval()
        self.switch_to_ema()
        self._setup_valid_data_generator()
        
        if self.config.ddp.global_rank == 0:
            self.logger.info(f"Generating samples for {self.config.generation.num_gen_samples} texts")

        # Generate samples
        total_num_texts = self.config.generation.num_gen_samples
        num_texts = int(total_num_texts / dist.get_world_size())
        if (total_num_texts % dist.get_world_size()) > dist.get_rank():
            num_texts += 1
        generated_sequences = self.generate_samples(num_texts)

        if self.config.ddp.enabled:
            generated_sequences = gather_texts(generated_sequences)

        # Save generated sequences
        if self.config.ddp.global_rank == 0:
            os.makedirs(self.config.generation.save_dir, exist_ok=True)
        
            prefix_folder = os.path.join(self.config.generation.save_dir, self.config.project.checkpoints_prefix)
            os.makedirs(prefix_folder, exist_ok=True)

            file_name = f"{self.step}.json"  
            save_path = os.path.join(prefix_folder, file_name)
            with open(save_path, "w") as f:
                json.dump(generated_sequences, f, indent=4)
            self.logger.info(f"Generated sequences are saved to {save_path}")

        # Reference sequences
        reference_sequences = []
        for batch in self.valid_loader:
            batch_size = min(len(batch["sequence"]), num_texts - len(reference_sequences))
            reference_sequences.extend(batch["sequence"][:batch_size])
            if len(reference_sequences) >= num_texts:
                break
        
        if self.config.ddp.enabled:
            reference_sequences = gather_texts(reference_sequences)

        # Calculate metrics
        result_metrics = {}
        for metric_name in self.config.metrics:
            num_samples = self.config.metrics[metric_name].num_samples
            max_len = self.config.datasets.max_sequence_len
            self.logger.info(f"Calculating {metric_name} for {num_samples} texts")
            
            tmp_result = compute_ddp_metric(
                metric_name,
                predictions=generated_sequences[:num_samples],
                references=reference_sequences[:num_samples],
                max_len = max_len,
                device = self.device,
                rank = self.config.ddp.global_rank,
                world_size = dist.get_world_size(),
                pdb_path = self.config.metrics[metric_name].pdb_path if "plddt" in metric_name else None
            )
            result_metrics[metric_name] = tmp_result
        
        # Logging
        if self.config.ddp.global_rank == 0:
            for key in result_metrics:
                value = result_metrics[key]
                self.logger.info(f"{key}: {value:0.5f}")
                log_metric("Metrics", key, value, self.step)

        self.switch_back_from_ema()
        self.ddp_score_estimator.train()

    def generate_samples(self, num_texts: int):
        results = []

        while len(results) < num_texts:
            batch_size = min(self.config.generation.batch_size, num_texts - len(results))

            # Generate text batch
            lens = self.length_sampler.sample(batch_size)
            attention_mask = self.encoder.get_attention_mask_for_lens(lens, max_sequence_len=self.config.datasets.max_sequence_len)
            with torch.no_grad():
                pred_embeddings = self.pred_embeddings(attention_mask)
                sequences = self.pred_logits(pred_embeddings, attention_mask=attention_mask)

            results.extend(sequences)

        return results

    def pred_embeddings(self, attention_mask: torch.Tensor):
        shape = (
            attention_mask.shape[0],
            attention_mask.shape[1],
            self.config.model.config.embedding_size
        )

        x = torch.randn(shape, device=self.device)
        x_0_self_cond = torch.zeros_like(x, dtype=x.dtype)
        eps_t = self.config.generation.t_min
        
        timesteps = torch.linspace(self.dynamic.T, eps_t, self.config.generation.N_steps + 1, device=self.device)
        for idx in tqdm(range(self.config.generation.N_steps)):
            t = timesteps[idx]
            next_t = timesteps[idx + 1]

            input_t = t * torch.ones(shape[0], device=self.device)
            next_input_t = next_t * torch.ones(shape[0], device=self.device)

            output = self.solver.step(
                x_t=x, t=input_t, next_t=next_input_t,
                mask=attention_mask,
                x_0_self_cond=x_0_self_cond,
            )
            x, x_mean = output["x"], output["x_mean"]
            x_0_self_cond = output["x_0"]
        return x_mean

    def pred_logits(self, pred_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        return self.encoder.batch_decode(encodings=pred_embeddings, attention_mask=attention_mask)
    
    def log_num_parameters(self):
        total_params = sum(p.numel() for p in self.score_estimator.parameters() if p.requires_grad)
        self.logger.info(f"Total number of parameters: {total_params}")
        log_metric("Statistics", "num_parameters", total_params, self.step)
