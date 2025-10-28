"""
骨架约束推理的 test_step 补丁
这个文件提供了修改后的 test_step 方法，支持动态读取目标分子式

使用方法：
    在 diffusion_model_spec2mol.py 的 test_step 方法中
    替换相应部分即可
"""

def test_step_with_dynamic_formula(self, batch, i):
    """
    修改后的 test_step，支持动态读取目标分子式
    """
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

    # 检查是否是推理模式
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

        true_E = torch.reshape(dense_data.E, (-1, dense_data.E.size(-1)))
        masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        self.test_CE(flat_pred_E, flat_true_E)

        true_mols = [Chem.inchi.MolFromInchi(data.get_example(idx).inchi) for idx in range(len(data))]
    else:
        # 推理模式：跳过loss计算
        true_mols = [None] * len(data)
    
    # ============ 骨架约束推理逻辑 ============
    use_scaffold = (
        hasattr(self.cfg.general, 'enforce_scaffold') and 
        self.cfg.general.enforce_scaffold and
        self.cfg.general.scaffold_smiles is not None
    )
    
    # 如果启用骨架约束，需要为每个样本设置 target_formula
    if use_scaffold:
        # 读取 labels 文件获取每个样本的分子式
        import pandas as pd
        
        if hasattr(self.cfg.dataset, 'labels_file') and self.cfg.dataset.labels_file:
            try:
                labels_df = pd.read_csv(self.cfg.dataset.labels_file, sep='\t')
                
                # 获取当前batch对应的spec_id
                # 假设 data 中有 spec_id 或类似字段
                # 如果没有，需要从batch index推导
                
                # 方法1: 如果 data 有 spec_id
                if hasattr(data, 'spec_id'):
                    spec_ids = data.spec_id
                    formulas = []
                    for spec_id in spec_ids:
                        row = labels_df[labels_df['spec'] == spec_id]
                        if len(row) > 0:
                            formulas.append(row.iloc[0]['formula'])
                        else:
                            formulas.append(None)
                            logging.warning(f"找不到 spec_id={spec_id} 的分子式，跳过骨架约束")
                
                # 方法2: 如果没有 spec_id，使用全局batch索引
                else:
                    # 这需要知道当前batch在整个数据集中的位置
                    # 暂时使用简单方法：假设按顺序处理
                    batch_size = len(data)
                    start_idx = i * batch_size  # i 是 batch_idx
                    
                    formulas = []
                    for local_idx in range(batch_size):
                        global_idx = start_idx + local_idx
                        if global_idx < len(labels_df):
                            formulas.append(labels_df.iloc[global_idx]['formula'])
                        else:
                            formulas.append(None)
                
                # 保存到配置中（临时）
                self.cfg.general._batch_formulas = formulas
                
            except Exception as e:
                logging.warning(f"无法读取labels文件: {e}，将不使用骨架约束")
                use_scaffold = False
    
    # 生成预测分子
    predicted_mols = [list() for _ in range(len(data))]
    
    for sample_idx in range(self.test_num_samples):
        if use_scaffold and hasattr(self.cfg.general, '_batch_formulas'):
            # 对每个样本分别进行骨架约束采样
            batch_mols = []
            
            for local_idx in range(len(data)):
                target_formula = self.cfg.general._batch_formulas[local_idx]
                
                if target_formula is None:
                    # 没有分子式，使用标准采样
                    single_data = self._extract_single_sample(data, local_idx)
                    single_mol_list = self.sample_batch(single_data)
                    batch_mols.append(single_mol_list[0] if single_mol_list else None)
                else:
                    # 使用骨架约束采样
                    single_data = self._extract_single_sample(data, local_idx)
                    
                    attachment_indices = getattr(self.cfg.general, 'attachment_indices', None)
                    if isinstance(attachment_indices, str):
                        attachment_indices = [int(x.strip()) for x in attachment_indices.split(',')]
                    
                    try:
                        single_mol_list = self.sample_batch_with_scaffold(
                            single_data,
                            scaffold_smiles=self.cfg.general.scaffold_smiles,
                            target_formula=target_formula,
                            attachment_indices=attachment_indices,
                            enforce_scaffold=True
                        )
                        batch_mols.append(single_mol_list[0] if single_mol_list else None)
                    except Exception as e:
                        logging.warning(f"骨架约束采样失败 (样本{local_idx}): {e}，回退到标准采样")
                        single_mol_list = self.sample_batch(single_data)
                        batch_mols.append(single_mol_list[0] if single_mol_list else None)
        else:
            # 使用标准批量采样
            batch_mols = self.sample_batch(data)
        
        for idx, mol in enumerate(batch_mols):
            predicted_mols[idx].append(mol)
    
    # 重排候选分子（如果启用）
    if hasattr(self.cfg.general, 'use_rerank') and self.cfg.general.use_rerank:
        from src.inference.rerank import rerank_by_spectrum, deduplicate_candidates
        import numpy as np
        
        for idx in range(len(predicted_mols)):
            if hasattr(batch, 'spectrum') and batch.spectrum is not None:
                spectrum_data = batch.spectrum
                if hasattr(spectrum_data, '__getitem__'):
                    spec_peaks = spectrum_data[idx]
                else:
                    spec_peaks = None
                
                if spec_peaks is not None:
                    unique_mols = deduplicate_candidates(predicted_mols[idx])
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
        with open(f"preds/{self.name}_rank_{self.global_rank}_true_{i}.pkl", "wb") as f:
            pickle.dump(true_mols, f)
        
        for idx in range(len(data)):
            self.test_k_acc.update(predicted_mols[idx], true_mols[idx])
            self.test_sim_metrics.update(predicted_mols[idx], true_mols[idx])
    
    for pred_mol_list in predicted_mols:
        self.test_validity.update(pred_mol_list)

    return {'loss': nll if not is_inference_mode else 0.0}


def _extract_single_sample(self, batch_data, idx):
    """
    从batch中提取单个样本
    
    这是一个辅助方法，需要根据实际的batch结构实现
    """
    # TODO: 根据实际的 batch 结构实现
    # 这里提供一个简化版本
    
    from torch_geometric.data import Batch, Data
    
    # 如果 batch_data 是 dict 形式
    if isinstance(batch_data, dict):
        single_graph = batch_data['graph'].get_example(idx)
        single_batch = Batch.from_data_list([single_graph])
        return {'graph': single_batch}
    else:
        # 如果是其他结构，需要相应调整
        return batch_data


# 使用说明:
"""
要应用这个补丁，需要在 diffusion_model_spec2mol.py 中：

1. 将原来的 test_step 方法替换为 test_step_with_dynamic_formula

2. 添加 _extract_single_sample 辅助方法

或者，更简单的方法：
直接修改现有的 test_step，在生成预测分子的部分，
读取 labels 文件并为每个样本设置正确的 target_formula

注意事项：
- 确保 cfg.dataset.labels_file 指向正确的 labels.tsv 文件
- 骨架约束采样是按样本逐个进行的，可能会比批量采样慢
- 如果某个样本的分子式与骨架不兼容，会自动回退到标准采样
"""

