import os
import time
import logging
import pickle
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import AllChem

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL, CrossEntropyMetric
from src.metrics.diffms_metrics import K_ACC_Collection, K_SimilarityCollection, Validity
from src import utils
from src.mist.models.spectra_encoder import SpectraEncoderGrowing
from src.inference import scaffold_hooks


class Spec2MolDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.decoder_dtype = torch.float32
        self.T = cfg.model.diffusion_steps
        self.val_num_samples = cfg.general.val_samples_to_generate
        self.test_num_samples = cfg.general.test_samples_to_generate

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_k_acc = K_ACC_Collection(list(range(1, self.val_num_samples + 1)))
        self.val_sim_metrics = K_SimilarityCollection(list(range(1, self.val_num_samples + 1)))
        self.val_validity = Validity()
        self.val_CE = CrossEntropyMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_k_acc = K_ACC_Collection(list(range(1, self.test_num_samples + 1)))
        self.test_sim_metrics = K_SimilarityCollection(list(range(1, self.test_num_samples + 1)))
        self.test_validity = Validity()
        self.test_CE = CrossEntropyMetric()

        self.train_metrics = train_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.decoder = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        try:
            if cfg.general.decoder is not None:
                state_dict = torch.load(cfg.general.decoder, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                    
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        k = k[6:]
                        cleaned_state_dict[k] = v

                self.decoder.load_state_dict(cleaned_state_dict)
        except Exception as e:
            logging.info(f"Could not load decoder: {e}")

        hidden_size = 256
        try:
            hidden_size = cfg.model.encoder_hidden_dim
        except:
            print("No hidden size specified, using default value of 256")

        magma_modulo = 512
        try:
            magma_modulo = cfg.model.encoder_magma_modulo
        except:
            print("No magma modulo specified, using default value of 512")
        
        self.encoder = SpectraEncoderGrowing(
                        inten_transform='float',
                        inten_prob=0.1,
                        remove_prob=0.5,
                        peak_attn_layers=2,
                        num_heads=8,
                        pairwise_featurization=True,
                        embed_instrument=False,
                        cls_type='ms1',
                        set_pooling='cls',
                        spec_features='peakformula',
                        mol_features='fingerprint',
                        form_embedder='pos-cos',
                        output_size=4096,
                        hidden_size=hidden_size,
                        spectra_dropout=0.1,
                        top_layers=1,
                        refine_layers=4,
                        magma_modulo=magma_modulo,
                    )
        
        try:
            if cfg.general.encoder is not None:
                self.encoder.load_state_dict(torch.load(cfg.general.encoder), strict=True)
        except Exception as e:
            logging.info(f"Could not load encoder: {e}")

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps)
        self.denoise_nodes = getattr(cfg.dataset, 'denoise_nodes', False)
        self.merge = getattr(cfg.dataset, 'merge', 'none')

        if self.merge == 'merge-encoder_output-linear':
            self.merge_function = nn.Linear(hidden_size, cfg.dataset.morgan_nbits)
        elif self.merge == 'merge-encoder_output-mlp':
            self.merge_function = nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.SiLU(),
                nn.Linear(1024, cfg.dataset.morgan_nbits)
            )
        elif self.merge == 'downproject_4096':
            self.merge_function = nn.Linear(4096, cfg.dataset.morgan_nbits)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            logging.info(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.best_val_nll = 1e8
        self.val_counter = 1
        
        # Atom and edge type vocabularies for scaffold-constrained generation
        self.atom_decoder = ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H']
        self.edge_decoder = ['no_edge', 'single', 'double', 'triple', 'aromatic']

    def training_step(self, batch, i):
        output, aux = self.encoder(batch)

        data = batch["graph"]
        if self.merge == 'mist_fp':
            data.y = aux["int_preds"][-1]
        if self.merge == 'merge-encoder_output-linear':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'merge-encoder_output-mlp':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'downproject_4096':
            data.y = self.merge_function(output)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=False)
 
        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=False)

        return {'loss': loss}

    def configure_optimizers(self):
        if self.cfg.train.scheduler == 'const':
            return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True, weight_decay=self.cfg.train.weight_decay)
        elif self.cfg.train.scheduler == 'one_cycle':
            opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True, weight_decay=self.cfg.train.weight_decay)
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.cfg.train.lr, total_steps=stepping_batches, pct_start=self.cfg.train.pct_start)
            lr_scheduler = {
                'scheduler': scheduler,
                'name': 'learning_rate',
                'interval':'step',
                'frequency': 1,
            }

            return [opt], [lr_scheduler]
        else:
            raise ValueError('Unknown Scheduler')

    def on_fit_start(self) -> None:
        if self.global_rank == 0:
            logging.info(f"Size of the input features: X-{self.Xdim}, E-{self.Edim}, y-{self.ydim}")
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        
    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        to_log["train_epoch/epoch"] = float(self.current_epoch)
        to_log["train_epoch/time"] = time.time() - self.start_epoch_time

        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        for key, value in epoch_at_metrics.items():
            to_log[f"train_epoch/{key}"] = value
        for key, value in epoch_bond_metrics.items():
            to_log[f"train_epoch/{key}"] = value

        self.log_dict(to_log, sync_dist=True)
        if self.global_rank == 0:
            logging.info(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE']:.2f} -- E_CE: {to_log['train_epoch/E_CE']:.2f} -- time: {to_log['train_epoch/time']:.2f}")

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_k_acc.reset()
        self.val_sim_metrics.reset()
        self.val_validity.reset()
        self.val_CE.reset()
        if self.global_rank == 0:
            self.val_counter += 1

    def validation_step(self, batch, i):
        output, aux = self.encoder(batch)

        data = batch["graph"]
        if self.merge == 'mist_fp':
            data.y = aux["int_preds"][-1]
        if self.merge == 'merge-encoder_output-linear':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'merge-encoder_output-mlp':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'downproject_4096':
            data.y = self.merge_function(output)


        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        pred = self.forward(noisy_data, extra_data, node_mask)
        pred.X = dense_data.X
        pred.Y = data.y

        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)

        true_E = torch.reshape(dense_data.E, (-1, dense_data.E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))   # (bs * n * n, de)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        self.val_CE(flat_pred_E, flat_true_E)

        if self.val_counter % self.cfg.general.sample_every_val == 0:
            true_mols = [Chem.inchi.MolFromInchi(data.get_example(idx).inchi) for idx in range(len(data))] # Is this correct?
            predicted_mols = [list() for _ in range(len(data))]
            for _ in range(self.val_num_samples):
                for idx, mol in enumerate(self.sample_batch(data)):
                    predicted_mols[idx].append(mol)
        
            for idx in range(len(data)):
                self.val_k_acc.update(predicted_mols[idx], true_mols[idx])
                self.val_sim_metrics.update(predicted_mols[idx], true_mols[idx])
                self.val_validity.update(predicted_mols[idx])

        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [
            self.val_nll.compute(), 
            self.val_X_kl.compute(), 
            self.val_E_kl.compute(),
            self.val_X_logp.compute(), 
            self.val_E_logp.compute(),
            self.val_CE.compute()
        ]

        log_dict = {
            "val/NLL": metrics[0],
            "val/X_KL": metrics[1],
            "val/E_KL": metrics[2],
            "val/X_logp": metrics[3],
            "val/E_logp": metrics[4],
            "val/E_CE": metrics[5]
        }

        if self.val_counter % self.cfg.general.sample_every_val == 0:
            for key, value in self.val_k_acc.compute().items():
                log_dict[f"val/{key}"] = value
            for key, value in self.val_sim_metrics.compute().items():
                log_dict[f"val/{key}"] = value
            log_dict["val/validity"] = self.val_validity.compute()

        self.log_dict(log_dict, sync_dist=True)

        if self.global_rank == 0:
            logging.info(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL: {metrics[1] :.2f} -- Val Edge type KL: {metrics[2] :.2f} -- Val Edge type logp: {metrics[4] :.2f} -- Val Edge type CE: {metrics[5] :.2f}")

            val_nll = metrics[0]
            if val_nll < self.best_val_nll:
                self.best_val_nll = val_nll
            logging.info(f"Val NLL: {val_nll :.4f} \t Best Val NLL:  {self.best_val_nll}")

    
    def on_test_epoch_start(self) -> None:
        if self.global_rank == 0:
            logging.info("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_k_acc.reset()
        self.test_sim_metrics.reset()
        self.test_validity.reset()
        self.test_CE.reset()

    def test_step(self, batch, i):
        output, aux = self.encoder(batch)

        data = batch["graph"]
        if self.merge == 'mist_fp':
            data.y = aux["int_preds"][-1]
        if self.merge == 'merge-encoder_output-linear':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'merge-encoder_output-mlp':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'downproject_4096':
            data.y = self.merge_function(output)

        # 检查是否是推理模式（没有真实分子数据）
        is_inference_mode = not hasattr(data, 'inchi') or data.inchi is None or len(getattr(data, 'inchi', [])) == 0
        
        if not is_inference_mode:
            # 训练/验证模式：计算loss和metrics
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
            extra_data = self.compute_extra_data(noisy_data)

            pred = self.forward(noisy_data, extra_data, node_mask)
            pred.X = dense_data.X
            pred.Y = data.y

            nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=True)

            true_E = torch.reshape(dense_data.E, (-1, dense_data.E.size(-1)))  # (bs * n * n, de)
            masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))   # (bs * n * n, de)
            mask_E = (true_E != 0.).any(dim=-1)

            flat_true_E = true_E[mask_E, :]
            flat_pred_E = masked_pred_E[mask_E, :]

            self.test_CE(flat_pred_E, flat_true_E)

            true_mols = [Chem.inchi.MolFromInchi(data.get_example(idx).inchi) for idx in range(len(data))]
        else:
            # 推理模式：跳过loss计算
            true_mols = [None] * len(data)  # 推理模式没有真实分子
        
        # 生成预测分子（训练/验证/推理模式都需要）
        # 检查是否使用骨架约束
        use_scaffold = (
            hasattr(self.cfg.general, 'enforce_scaffold') and 
            self.cfg.general.enforce_scaffold and
            self.cfg.general.scaffold_smiles is not None
        )
        
        # 读取每个样本的 formula（如果使用骨架约束）
        batch_formulas = None
        if use_scaffold and hasattr(self.cfg.dataset, 'labels_file'):
            try:
                import pandas as pd
                labels_df = pd.read_csv(self.cfg.dataset.labels_file, sep='\t')
                
                # 提取当前 batch 的 formulas
                batch_size = len(data)
                batch_formulas = []
                
                # 方法：使用 batch 索引推导（假设按顺序处理）
                # 每个 test_step 调用对应一个 batch
                start_idx = i * batch_size  # i 是 batch 索引
                
                for local_idx in range(batch_size):
                    global_idx = start_idx + local_idx
                    if global_idx < len(labels_df):
                        formula = labels_df.iloc[global_idx]['formula']
                        batch_formulas.append(formula)
                    else:
                        batch_formulas.append(None)
                        logging.warning(f"Sample {local_idx} in batch {i} has no formula, using standard sampling")
                
                logging.info(f"Batch {i}: loaded {len([f for f in batch_formulas if f])} formulas")
                
            except Exception as e:
                logging.warning(f"Failed to load formulas from labels file: {e}, using standard sampling")
                use_scaffold = False
                batch_formulas = None
        
        predicted_mols = [list() for _ in range(len(data))]
        for _ in range(self.test_num_samples):
            if use_scaffold and batch_formulas:
                # 使用骨架约束采样（批量模式）
                attachment_indices = getattr(self.cfg.general, 'attachment_indices', None)
                if isinstance(attachment_indices, str):
                    attachment_indices = [int(x.strip()) for x in attachment_indices.split(',')]
                
                batch_mols = self.sample_batch_with_scaffold(
                    data,
                    scaffold_smiles=self.cfg.general.scaffold_smiles,
                    target_formula=batch_formulas,  # 传入列表
                    attachment_indices=attachment_indices,
                    enforce_scaffold=True
                )
            else:
                # 使用标准采样
                batch_mols = self.sample_batch(data)
            
            for idx, mol in enumerate(batch_mols):
                predicted_mols[idx].append(mol)
        
        # 重排候选分子（如果启用）
        if hasattr(self.cfg.general, 'use_rerank') and self.cfg.general.use_rerank:
            from src.inference.rerank import rerank_by_spectrum, deduplicate_candidates
            import numpy as np
            
            # 对每个样本的候选分子进行重排
            for idx in range(len(predicted_mols)):
                # 获取对应的质谱数据
                # 注意：这里需要根据实际的batch结构来提取质谱峰数据
                # 假设batch中有spectrum字段
                if hasattr(batch, 'spectrum') and batch.spectrum is not None:
                    # 假设spectrum是字典或对象，包含peaks数据
                    spectrum_data = batch.spectrum
                    if hasattr(spectrum_data, '__getitem__'):
                        spec_peaks = spectrum_data[idx]  # 获取该样本的谱峰
                    else:
                        spec_peaks = None
                    
                    if spec_peaks is not None:
                        # 去重
                        unique_mols = deduplicate_candidates(predicted_mols[idx])
                        # 重排
                        reranked_mols = rerank_by_spectrum(
                            unique_mols,
                            spec_peaks,
                            top_k_pre=min(64, len(unique_mols)),
                            use_accurate_rerank=False
                        )
                        predicted_mols[idx] = reranked_mols

        # 保存预测结果
        with open(f"preds/{self.name}_rank_{self.global_rank}_pred_{i}.pkl", "wb") as f:
            pickle.dump(predicted_mols, f)
        
        if not is_inference_mode:
            # 只有在有真实分子时才保存和计算metrics
            with open(f"preds/{self.name}_rank_{self.global_rank}_true_{i}.pkl", "wb") as f:
                pickle.dump(true_mols, f)
            
            for idx in range(len(data)):
                self.test_k_acc.update(predicted_mols[idx], true_mols[idx])
                self.test_sim_metrics.update(predicted_mols[idx], true_mols[idx])
        
        # 更新validity metrics（对所有模式）
        for pred_mol_list in predicted_mols:
            self.test_validity.update(pred_mol_list)

        # 返回loss（推理模式下为0）
        return {'loss': nll if not is_inference_mode else 0.0}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [
            self.test_nll.compute(), 
            self.test_X_kl.compute(), 
            self.test_E_kl.compute(),
            self.test_X_logp.compute(), 
            self.test_E_logp.compute(),
            self.test_CE.compute()
        ]

        log_dict = {
            "test/NLL": metrics[0],
            "test/X_KL": metrics[1],
            "test/E_KL": metrics[2],
            "test/X_logp": metrics[3],
            "test/E_logp": metrics[4],
            "test/E_CE": metrics[5]
        }

        self.log_dict(log_dict, sync_dist=True)
        if self.global_rank == 0:
            logging.info(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- Test Edge type KL: {metrics[2] :.2f} -- Test Edge type logp: {metrics[3] :.2f} -- Test Edge type CE: {metrics[5] :.2f}")

        log_dict = {}
        for key, value in self.test_k_acc.compute().items():
            log_dict[f"test/{key}"] = value
        for key, value in self.test_sim_metrics.compute().items():
            log_dict[f"test/{key}"] = value
        log_dict["test/validity"] = self.test_validity.compute()

        self.log_dict(log_dict, sync_dist=True)
        
        
    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, y, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        lowest_t = 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = X
        if self.denoise_nodes:
            X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
        """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, y, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        if test:
            nll = self.test_nll(nlls)
        else:
            nll = self.val_nll(nlls)

        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.decoder(X, E, y, node_mask)
    
    @torch.no_grad()
    def sample_batch(self, data: Batch) -> Batch:
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)

        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = dense_data.X, z_T.E, data.y

        assert (E == torch.transpose(E, 1, 2)).all()

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((len(data), 1), dtype=torch.float32, device=self.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, __ = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            _, E, y = sampled_s.X, sampled_s.E, data.y

        # Sample
        sampled_s.X = X
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, data.y

        mols = []
        for nodes, adj_mat in zip(X, E):
            mol = self.visualization_tools.mol_from_graphs(nodes, adj_mat)
            
            # 关键修复：应用价态修正（与论文一致）
            if mol is not None:
                from rdkit import Chem
                from analysis.rdkit_functions import correct_mol
                try:
                    # 转换为RWMol（可编辑）
                    editable_mol = Chem.RWMol(mol)
                    corrected_mol, no_correct = correct_mol(editable_mol)
                    if corrected_mol is not None:
                        mol = corrected_mol
                    # 如果correct_mol返回None，保留原分子
                except Exception as e:
                    # 修正失败，保留原分子
                    import logging
                    logging.debug(f"Molecule correction failed: {e}")
            
            mols.append(mol)

        return mols

    @torch.no_grad()
    def sample_batch_with_scaffold(
        self, 
        data: Batch,
        scaffold_smiles: str,
        target_formula: str | list[str],  # 支持单个或批量
        attachment_indices: list[int] = None,
        enforce_scaffold: bool = True
    ) -> Batch:
        """
        Scaffold-constrained sampling with formula and attachment point constraints.
        
        Args:
            data: Batch data containing spectrum embeddings
            scaffold_smiles: SMILES string of the scaffold substructure
            target_formula: Target molecular formula (e.g., 'C10H12N2O')
            attachment_indices: List of atom indices in scaffold where new fragments can attach
            enforce_scaffold: If True, strictly enforce scaffold presence
        
        Returns:
            List of generated molecules
        """
        # Parse scaffold
        scaffold_mol = scaffold_hooks.smiles_to_mol(scaffold_smiles)
        scaffold_f = scaffold_hooks.formula_of(scaffold_mol)
        
        # 支持批量 formula（每个样本一个）
        if isinstance(target_formula, list):
            # 批量模式：每个样本分别处理
            # 正确计算批次大小：使用 num_graphs 或 batch 属性
            if hasattr(data, 'num_graphs'):
                batch_size = data.num_graphs
            elif hasattr(data, 'batch'):
                batch_size = data.batch.max().item() + 1
            else:
                batch_size = 1
            
            if len(target_formula) != batch_size:
                raise ValueError(f"Formula list length ({len(target_formula)}) != batch size ({batch_size})")
            
            # 对每个样本分别采样（逐个处理）
            all_mols = []
            for idx in range(batch_size):
                single_data = self._extract_single_from_batch(data, idx)
                single_formula = target_formula[idx]
                
                # 调试：检查提取的数据是否包含 y
                if not hasattr(single_data, 'y') or single_data.y is None:
                    logging.error(f"Sample {idx}: Extracted data missing y (spectrum embedding)")
                    logging.error(f"  single_data type: {type(single_data)}")
                    logging.error(f"  single_data attributes: {dir(single_data)}")
                    # 尝试回退到标准采样
                    single_mols = self.sample_batch(single_data)
                    all_mols.append(single_mols[0] if single_mols else None)
                    continue
                
                try:
                    single_mols = self.sample_batch_with_scaffold(
                        single_data,
                        scaffold_smiles=scaffold_smiles,
                        target_formula=single_formula,  # 单个 formula
                        attachment_indices=attachment_indices,
                        enforce_scaffold=enforce_scaffold
                    )
                    all_mols.append(single_mols[0] if single_mols else None)
                except Exception as e:
                    logging.warning(f"Sample {idx} scaffold sampling failed: {e}, using standard sampling")
                    single_mols = self.sample_batch(single_data)
                    all_mols.append(single_mols[0] if single_mols else None)
            
            return all_mols
        
        # 单个 formula 模式（原逻辑）
        target_f = scaffold_hooks.parse_formula(target_formula)
        
        # Calculate remaining formula (ΔF = target - scaffold)
        try:
            remaining_f = scaffold_hooks.formula_subtract(target_f, scaffold_f)
        except ValueError as e:
            logging.warning(f"Formula constraint violated: {e}. Scaffold requires more atoms than target formula.")
            # Fallback: use normal sampling
            return self.sample_batch(data)
        
        logging.info(f"Scaffold formula: {scaffold_hooks.formula_to_string(scaffold_f)}")
        logging.info(f"Target formula: {scaffold_hooks.formula_to_string(target_f)}")
        logging.info(f"Remaining formula (ΔF): {scaffold_hooks.formula_to_string(remaining_f)}")
        
        # 关键检查：确保 data.y 存在
        if not hasattr(data, 'y') or data.y is None:
            logging.error("Data missing y (spectrum embedding), cannot proceed with scaffold sampling")
            logging.error(f"  data type: {type(data)}")
            logging.error(f"  data attributes: {[attr for attr in dir(data) if not attr.startswith('_')]}")
            raise ValueError("Spectrum embedding (y) is required for scaffold-constrained sampling")
        
        # Initialize dense graph
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # Start from COMPLETE NOISE (both X and E) to match model's expectation at t=T
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, data.y  # ← 修复：X也用噪声，而不是dense_data.X
        
        # Prepare scaffold metadata (DON'T modify X/E at t=T - keep pure noise)
        scaffold_size = scaffold_mol.GetNumAtoms()
        scaffold_indices = list(range(min(scaffold_size, X.shape[1])))
        
        logging.info(f"[DEBUG] Scaffold-constrained sampling:")
        logging.info(f"  Scaffold size: {scaffold_size} atoms, {scaffold_mol.GetNumBonds()} bonds")
        logging.info(f"  Starting from PURE NOISE at t=T")
        logging.info(f"  HOOK 3 will enforce scaffold during reverse diffusion")
        
        if False and enforce_scaffold and scaffold_size <= X.shape[1]:  # DISABLED initialization
            # A. Overwrite scaffold atoms in X
            logging.info(f"[DEBUG] Initializing scaffold atoms and bonds:")
            for local_idx in range(scaffold_size):
                if local_idx >= X.shape[1]:
                    break
                atom = scaffold_mol.GetAtomWithIdx(local_idx)
                atom_symbol = atom.GetSymbol()
                if atom_symbol in self.atom_decoder:
                    atom_type_idx = self.atom_decoder.index(atom_symbol)
                    X[:, local_idx, :] = 0
                    X[:, local_idx, atom_type_idx] = 1
                    if local_idx < 5:  # 只打印前5个
                        logging.info(f"  Node {local_idx}: set to {atom_symbol} (idx={atom_type_idx})")
            
            # B. Overwrite scaffold bonds in E
            from rdkit import Chem
            bond_count = 0
            for bond in scaffold_mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Only set if both atoms are within the graph size
                if i >= E.shape[1] or j >= E.shape[1]:
                    continue
                
                # Get bond type
                bond_type = bond.GetBondType()
                if bond_type == Chem.BondType.SINGLE:
                    edge_type_idx = 0
                elif bond_type == Chem.BondType.DOUBLE:
                    edge_type_idx = 1
                elif bond_type == Chem.BondType.TRIPLE:
                    edge_type_idx = 2
                elif bond_type == Chem.BondType.AROMATIC:
                    edge_type_idx = 3
                else:
                    edge_type_idx = 0  # default to single
                
                # Set both directions (symmetric)
                E[:, i, j, :] = 0
                E[:, i, j, edge_type_idx] = 1
                E[:, j, i, :] = 0
                E[:, j, i, edge_type_idx] = 1
                bond_count += 1
                
                if bond_count <= 5:  # 只打印前5个
                    bond_type_name = str(bond_type).split('.')[-1]
                    logging.info(f"  Bond {i}-{j}: {bond_type_name} (idx={edge_type_idx})")
            
            logging.info(f"  Total: {scaffold_size} atoms, {bond_count} bonds initialized")
            
            # 验证边初始化
            logging.info(f"[DEBUG] Verifying edge initialization:")
            edge_verify_count = 0
            for bond in scaffold_mol.GetBonds():
                if edge_verify_count >= 3:
                    break
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if i >= E.shape[1] or j >= E.shape[1]:
                    continue
                
                edge_types = E[0, i, j, :]
                predicted_edge = torch.argmax(edge_types).item()
                
                bond_type = bond.GetBondType()
                if bond_type == Chem.BondType.SINGLE:
                    expected_edge = 0
                elif bond_type == Chem.BondType.DOUBLE:
                    expected_edge = 1
                elif bond_type == Chem.BondType.TRIPLE:
                    expected_edge = 2
                elif bond_type == Chem.BondType.AROMATIC:
                    expected_edge = 3
                else:
                    expected_edge = 0
                
                match = "✓" if predicted_edge == expected_edge else "✗"
                logging.info(f"  Edge {i}-{j}: type {predicted_edge} (expected: {expected_edge}) {match}")
                edge_verify_count += 1
            
            # 验证原子初始化
            logging.info(f"[DEBUG] Verifying atom initialization:")
            for local_idx in range(min(5, scaffold_size)):
                atom_types = X[0, local_idx, :]
                predicted_type = torch.argmax(atom_types).item()
                predicted_symbol = self.atom_decoder[predicted_type]
                expected_symbol = scaffold_mol.GetAtomWithIdx(local_idx).GetSymbol()
                match = "✓" if predicted_symbol == expected_symbol else "✗"
                logging.info(f"  Node {local_idx}: {predicted_symbol} (expected: {expected_symbol}) {match}")
        
        assert (E == torch.transpose(E, 1, 2)).all()
        
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((len(data), 1), dtype=torch.float32, device=self.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            
            # Sample z_s with scaffold constraints
            sampled_s, __ = self.sample_p_zs_given_zt_with_scaffold(
                s_norm, t_norm, X, E, y, node_mask,
                scaffold_mol=scaffold_mol,
                remaining_formula=remaining_f,
                scaffold_indices=scaffold_indices,
                attachment_indices=attachment_indices
            )
            # 关键修复：更新 X 以使骨架冻结生效
            X, E, y = sampled_s.X, sampled_s.E, data.y
            
            # 每100步检查一次
            if s_int % 100 == 0:
                logging.info(f"[DEBUG] Step {s_int}: Checking scaffold preservation...")
                
                # 检查原子
                atom_mismatch = 0
                for local_idx in range(min(scaffold_size, X.shape[1])):
                    atom_types = X[0, local_idx, :]
                    predicted_type = torch.argmax(atom_types).item()
                    predicted_symbol = self.atom_decoder[predicted_type]
                    expected_symbol = scaffold_mol.GetAtomWithIdx(local_idx).GetSymbol()
                    if predicted_symbol != expected_symbol:
                        atom_mismatch += 1
                        if atom_mismatch <= 3:  # 只打印前3个不匹配
                            logging.warning(f"  ✗ Atom {local_idx}: {predicted_symbol} != {expected_symbol}")
                
                if atom_mismatch == 0:
                    logging.info(f"  ✓ All {scaffold_size} atoms match")
                else:
                    logging.warning(f"  ✗ {atom_mismatch}/{scaffold_size} atoms mismatch!")
                
                # 检查边
                edge_mismatch = 0
                edge_total = 0
                from rdkit import Chem
                for bond in scaffold_mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    if i >= E.shape[1] or j >= E.shape[1]:
                        continue
                    
                    edge_types = E[0, i, j, :]
                    predicted_edge = torch.argmax(edge_types).item()
                    
                    bond_type = bond.GetBondType()
                    if bond_type == Chem.BondType.SINGLE:
                        expected_edge = 0
                    elif bond_type == Chem.BondType.DOUBLE:
                        expected_edge = 1
                    elif bond_type == Chem.BondType.TRIPLE:
                        expected_edge = 2
                    elif bond_type == Chem.BondType.AROMATIC:
                        expected_edge = 3
                    else:
                        expected_edge = 0
                    
                    edge_total += 1
                    if predicted_edge != expected_edge:
                        edge_mismatch += 1
                        if edge_mismatch <= 3:  # 只打印前3个不匹配
                            logging.warning(f"  ✗ Edge {i}-{j}: type {predicted_edge} != {expected_edge}")
                
                if edge_mismatch == 0:
                    logging.info(f"  ✓ All {edge_total} edges match")
                else:
                    logging.warning(f"  ✗ {edge_mismatch}/{edge_total} edges mismatch!")
        
        # Final sampling (X already updated in loop)
        logging.info(f"[DEBUG] After diffusion loop, final verification:")
        
        # 检查原子
        atom_mismatch = 0
        for local_idx in range(min(5, scaffold_size)):
            atom_types = X[0, local_idx, :]
            predicted_type = torch.argmax(atom_types).item()
            predicted_symbol = self.atom_decoder[predicted_type]
            expected_symbol = scaffold_mol.GetAtomWithIdx(local_idx).GetSymbol()
            match = "✓" if predicted_symbol == expected_symbol else "✗"
            if predicted_symbol != expected_symbol:
                atom_mismatch += 1
            logging.info(f"  Node {local_idx}: {predicted_symbol} (expected: {expected_symbol}) {match}")
        
        # 检查边
        from rdkit import Chem
        edge_mismatch = 0
        edge_verify_count = 0
        for bond in scaffold_mol.GetBonds():
            if edge_verify_count >= 5:
                break
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i >= E.shape[1] or j >= E.shape[1]:
                continue
            
            edge_types = E[0, i, j, :]
            predicted_edge = torch.argmax(edge_types).item()
            
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                expected_edge = 0
            elif bond_type == Chem.BondType.DOUBLE:
                expected_edge = 1
            elif bond_type == Chem.BondType.TRIPLE:
                expected_edge = 2
            elif bond_type == Chem.BondType.AROMATIC:
                expected_edge = 3
            else:
                expected_edge = 0
            
            match = "✓" if predicted_edge == expected_edge else "✗"
            if predicted_edge != expected_edge:
                edge_mismatch += 1
            logging.info(f"  Edge {i}-{j}: type {predicted_edge} (expected: {expected_edge}) {match}")
            edge_verify_count += 1
        
        if atom_mismatch > 0 or edge_mismatch > 0:
            logging.warning(f"[CRITICAL] Scaffold not preserved! Atoms: {atom_mismatch} mismatch, Edges: {edge_mismatch} mismatch")
        
        # ===== 关键修复：绕过mask，直接使用我们维护的X和E =====
        if enforce_scaffold:
            logging.info(f"[DEBUG] Bypassing mask to preserve scaffold edges")
            logging.info(f"  X.shape before processing: {X.shape}, E.shape: {E.shape}")
            
            # 直接从4D张量转换为用于mol_from_graphs的格式
            # X: [1, n_nodes, atom_types] -> 需要转换为 [batch, n_nodes]（argmax后）
            # E: [1, n_nodes, n_nodes, edge_types] -> 需要转换为 [batch, n_nodes, n_nodes]（argmax后）
            
            # 对X做argmax得到atom type indices
            X_indices = torch.argmax(X, dim=-1)  # [1, n_nodes]
            
            # 对E做argmax得到edge type indices
            E_indices = torch.argmax(E, dim=-1)  # [1, n_nodes, n_nodes]
            
            # 验证骨架边是否保留
            edge_check = []
            for i in range(min(10, E_indices.shape[1])):
                for j in range(i+1, min(10, E_indices.shape[2])):
                    edge_type = E_indices[0, i, j].item()
                    if edge_type < 4:  # 不是NO_EDGE
                        edge_check.append(f"{i}-{j}:type{edge_type}")
            logging.info(f"  Edges preserved (first 10): {edge_check}")
            
            # 使用处理后的X和E
            X = X_indices
            E = E_indices
            y = data.y
        else:
            # 原始路径：使用mask
            sampled_s.X = X
            sampled_s = sampled_s.mask(node_mask, collapse=True)
            X, E, y = sampled_s.X, sampled_s.E, data.y
        
        mols = []
        for mol_idx, (nodes, adj_mat) in enumerate(zip(X, E)):
            # 调试：打印节点和边信息（只打印第一个分子）
            if mol_idx == 0 and enforce_scaffold:
                logging.info(f"[DEBUG] Converting graph #{mol_idx} to molecule:")
                logging.info(f"  nodes.shape = {nodes.shape}")
                logging.info(f"  adj_mat.shape = {adj_mat.shape}")
                
                # 打印前10个节点的类型
                node_types = []
                if len(nodes.shape) == 1:  # 1D: [n]
                    for i in range(min(10, nodes.shape[0])):
                        node_type = nodes[i].item()  # 直接取值
                        node_symbol = self.atom_decoder[node_type]
                        node_types.append(node_symbol)
                elif len(nodes.shape) == 2:  # 2D: [n, atom_types]
                    for i in range(min(10, nodes.shape[0])):
                        node_type = torch.argmax(nodes[i]).item()
                        node_symbol = self.atom_decoder[node_type]
                        node_types.append(node_symbol)
                logging.info(f"  First 10 node types: {node_types}")
                
                # 统计边类型
                edge_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                if len(adj_mat.shape) == 2:  # 2D: [n, n]
                    for i in range(adj_mat.shape[0]):
                        for j in range(i+1, adj_mat.shape[1]):
                            edge_type = adj_mat[i, j].item()  # 直接取值，不用argmax
                            edge_counts[edge_type] += 1
                elif len(adj_mat.shape) == 3:  # 3D: [n, n, edge_types]
                    for i in range(adj_mat.shape[0]):
                        for j in range(i+1, adj_mat.shape[1]):
                            edge_type = torch.argmax(adj_mat[i, j, :]).item()
                            edge_counts[edge_type] += 1
                logging.info(f"  Edge type counts: SINGLE={edge_counts[0]}, DOUBLE={edge_counts[1]}, TRIPLE={edge_counts[2]}, AROMATIC={edge_counts[3]}, NO_EDGE={edge_counts[4]}")
            
            mol = self.visualization_tools.mol_from_graphs(nodes, adj_mat)
            
            # 调试：在价态校正前检查分子
            if mol_idx == 0 and enforce_scaffold and mol is not None:
                from rdkit import Chem
                try:
                    logging.info(f"[DEBUG] After mol_from_graphs (before valence correction):")
                    logging.info(f"  Mol has {mol.GetNumAtoms()} atoms")
                    mol_smiles = Chem.MolToSmiles(mol)
                    logging.info(f"  Mol SMILES: {mol_smiles[:100]}...")
                except Exception as e:
                    logging.error(f"[DEBUG] Error getting mol info: {e}")
            
            # Apply valence correction
            if mol is not None:
                from rdkit import Chem
                from analysis.rdkit_functions import correct_mol
                try:
                    editable_mol = Chem.RWMol(mol)
                    corrected_mol, no_correct = correct_mol(editable_mol)
                    if corrected_mol is not None:
                        mol = corrected_mol
                        
                        # 调试：价态校正后
                        if mol_idx == 0 and enforce_scaffold:
                            logging.info(f"[DEBUG] After valence correction:")
                            corrected_smiles = Chem.MolToSmiles(mol)
                            logging.info(f"  Corrected has {mol.GetNumAtoms()} atoms")
                            logging.info(f"  Corrected SMILES: {corrected_smiles[:100]}...")
                except Exception as e:
                    logging.debug(f"Molecule correction failed: {e}")
            
            # Validate scaffold presence (if enforce_scaffold)
            if enforce_scaffold and mol is not None:
                # 调试：打印生成的分子信息
                try:
                    gen_smiles = Chem.MolToSmiles(mol)
                    scaf_smiles = Chem.MolToSmiles(scaffold_mol)
                    logging.info(f"[DEBUG] Generated mol: {gen_smiles[:100]}...")
                    logging.info(f"[DEBUG] Scaffold: {scaf_smiles[:100]}...")
                    logging.info(f"[DEBUG] Generated has {mol.GetNumAtoms()} atoms, scaffold has {scaffold_mol.GetNumAtoms()} atoms")
                except Exception as e:
                    logging.error(f"[DEBUG] Error getting SMILES: {e}")
                
                if not scaffold_hooks.contains_scaffold(mol, scaffold_mol):
                    logging.warning("Generated molecule does not contain scaffold. Discarding.")
                    mol = None
                else:
                    logging.info(f"✅ Generated molecule CONTAINS scaffold!")
            
            mols.append(mol)
        
        return mols

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def sample_p_zs_given_zt_with_scaffold(
        self, s, t, X_t, E_t, y_t, node_mask,
        scaffold_mol=None,
        remaining_formula=None,
        scaffold_indices=None,
        attachment_indices=None
    ):
        """
        Samples from zs ~ p(zs | zt) with scaffold and formula constraints.
        
        Args:
            s, t: Time steps
            X_t, E_t, y_t: Current graph state
            node_mask: Node mask
            scaffold_mol: RDKit molecule for scaffold
            remaining_formula: Remaining formula (ΔF) after scaffold
            scaffold_indices: Indices of scaffold atoms in the graph
            attachment_indices: Allowed attachment points
        
        Returns:
            Sampled next state with constraints applied
        """
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        
        # Retrieve transition matrices
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)
        
        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        
        # === HOOK 1: Apply formula mask to node logits ===
        if remaining_formula is not None and scaffold_indices is not None:
            # Apply formula mask only to non-scaffold nodes
            pred_X_masked = pred.X.clone()
            for node_idx in range(n):
                if node_idx not in scaffold_indices:
                    # Apply formula constraint
                    pred_X_masked[:, node_idx, :] = scaffold_hooks.apply_formula_mask_to_logits(
                        pred.X[:, node_idx:node_idx+1, :],
                        remaining_formula,
                        self.atom_decoder
                    )[:, 0, :]
            pred.X = pred_X_masked
        
        # Normalize predictions (softmax)
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        
        # === HOOK 2: Apply attachment mask to edge logits (optional) ===
        # Note: This is simplified - in practice you might want more sophisticated edge masking
        # For now, we rely on the scaffold freeze to handle most of the constraint
        
        # Compute posterior distributions
        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(
            X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
        )
        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(
            X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
        )
        
        # Weighted probabilities
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)
        
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
        
        # === HOOK 3: Freeze scaffold atoms and bonds ===
        if scaffold_mol is not None and scaffold_indices is not None:
            frozen_atom_count = 0
            frozen_bond_count = 0
            
            # 3A: Freeze atom types (nodes)
            for local_idx in scaffold_indices:
                if local_idx >= scaffold_mol.GetNumAtoms() or local_idx >= n:
                    continue
                atom = scaffold_mol.GetAtomWithIdx(local_idx)
                atom_symbol = atom.GetSymbol()
                if atom_symbol in self.atom_decoder:
                    atom_type_idx = self.atom_decoder.index(atom_symbol)
                    # Force scaffold atoms to stay fixed
                    prob_X[:, local_idx, :] = 0
                    prob_X[:, local_idx, atom_type_idx] = 1
                    frozen_atom_count += 1
            
            # 3B: CRITICAL - First set all scaffold internal edges to NO_EDGE
            num_scaffold_atoms = scaffold_mol.GetNumAtoms()
            NO_EDGE_idx = 4  # NO_EDGE type
            for i in range(min(num_scaffold_atoms, n)):
                for j in range(min(num_scaffold_atoms, n)):
                    if i != j:  # No self-loops
                        prob_E[:, i, j, :] = 0
                        prob_E[:, i, j, NO_EDGE_idx] = 1
            
            # 3C: Then freeze scaffold bonds with their actual types
            for bond in scaffold_mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Only freeze if both atoms are within the graph size
                if i >= n or j >= n:
                    continue
                
                # Get bond type
                bond_type = bond.GetBondType()
                from rdkit import Chem
                if bond_type == Chem.BondType.SINGLE:
                    edge_type_idx = 0
                elif bond_type == Chem.BondType.DOUBLE:
                    edge_type_idx = 1
                elif bond_type == Chem.BondType.TRIPLE:
                    edge_type_idx = 2
                elif bond_type == Chem.BondType.AROMATIC:
                    edge_type_idx = 3
                else:
                    edge_type_idx = 0  # default to single
                
                # Freeze both directions (since adjacency matrix is symmetric)
                prob_E[:, i, j, :] = 0
                prob_E[:, i, j, edge_type_idx] = 1
                prob_E[:, j, i, :] = 0
                prob_E[:, j, i, edge_type_idx] = 1
                frozen_bond_count += 1
            
            # 每100步打印一次
            if hasattr(t, '__getitem__'):
                t_val = t[0, 0].item()
            else:
                t_val = t.item() if hasattr(t, 'item') else float(t)
            
            if int(t_val * 500) % 100 == 0:  # 假设 T=500
                logging.info(f"[HOOK 3] Frozen {frozen_atom_count} atoms, {frozen_bond_count} bonds at t={t_val:.3f}")
        
        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()
        
        # Sample discrete features
        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)
        
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()
        
        # === POST-SAMPLING HOOK: Force replace scaffold after sampling ===
        if scaffold_mol is not None and scaffold_indices is not None:
            # Overwrite scaffold atoms in X_s (after sampling)
            for local_idx in scaffold_indices:
                if local_idx >= scaffold_mol.GetNumAtoms() or local_idx >= n:
                    continue
                atom = scaffold_mol.GetAtomWithIdx(local_idx)
                atom_symbol = atom.GetSymbol()
                if atom_symbol in self.atom_decoder:
                    atom_type_idx = self.atom_decoder.index(atom_symbol)
                    X_s[:, local_idx, :] = 0
                    X_s[:, local_idx, atom_type_idx] = 1
            
            # CRITICAL: First set all scaffold internal edges to NO_EDGE (type 4)
            # This prevents random edges between scaffold atoms
            num_scaffold_atoms = scaffold_mol.GetNumAtoms()
            NO_EDGE_idx = 4  # NO_EDGE type
            for i in range(min(num_scaffold_atoms, n)):
                for j in range(min(num_scaffold_atoms, n)):
                    if i != j:  # No self-loops
                        E_s[:, i, j, :] = 0
                        E_s[:, i, j, NO_EDGE_idx] = 1
            
            # Then overwrite scaffold bonds with their actual types
            for bond in scaffold_mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                if i >= n or j >= n:
                    continue
                
                bond_type = bond.GetBondType()
                from rdkit import Chem
                if bond_type == Chem.BondType.SINGLE:
                    edge_type_idx = 0
                elif bond_type == Chem.BondType.DOUBLE:
                    edge_type_idx = 1
                elif bond_type == Chem.BondType.TRIPLE:
                    edge_type_idx = 2
                elif bond_type == Chem.BondType.AROMATIC:
                    edge_type_idx = 3
                else:
                    edge_type_idx = 0
                
                E_s[:, i, j, :] = 0
                E_s[:, i, j, edge_type_idx] = 1
                E_s[:, j, i, :] = 0
                E_s[:, j, i, edge_type_idx] = 1
        
        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)
        
        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        
        # CRITICAL: Bypass mask when scaffold is enforced to preserve our carefully set edges
        if scaffold_mol is not None and scaffold_indices is not None:
            # Don't apply mask - return raw tensors to preserve scaffold edges
            return out_one_hot.type_as(y_t), out_discrete.type_as(y_t)
        else:
            # Original path: apply mask
            return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def _extract_single_from_batch(self, batch_data, idx: int):
        """从 batch 中提取单个样本，保留所有字段包括 y（谱嵌入）"""
        from torch_geometric.data import Batch, Data
        
        # 如果是dict形式的batch
        if isinstance(batch_data, dict) and 'graph' in batch_data:
            single_graph = batch_data['graph'].get_example(idx)
            single_batch = Batch.from_data_list([single_graph])
            # 提取对应的 y（谱嵌入）
            if hasattr(batch_data['graph'], 'y') and batch_data['graph'].y is not None:
                single_batch.y = batch_data['graph'].y[idx:idx+1]
            return {'graph': single_batch}
        else:
            # 直接是 Batch 对象
            single_graph = batch_data.get_example(idx)
            single_batch = Batch.from_data_list([single_graph])
            # 关键修复：保留 y（谱嵌入数据）
            if hasattr(batch_data, 'y') and batch_data.y is not None:
                single_batch.y = batch_data.y[idx:idx+1].clone()
            return single_batch

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
