# 🔥 关键修复：绕过mask保留骨架边

**时间**: 2025-10-28  
**问题**: `mask(collapse=True)` 破坏了骨架的边信息  
**解决**: 完全绕过mask，直接使用argmax

---

## 💥 问题根源

### mask前（正确）✅
```
First few edges: ['0-1:type0', '0-2:type1', '1-2:type1', ...]
                        ↑ 单键      ↑ 双键      ↑ 骨架双键
```

### mask后（错误）❌
```
Edges after mask: ['0-1:type2', '0-2:type0', ...]
                        ↑ 变成三键！  ↑ 1-2双键消失
```

**`mask(collapse=True)` 的破坏性**：
1. 边类型错误（type0→type2）
2. 边索引混乱（1-2→0-2）
3. 关键双键丢失

---

## 🔧 解决方案

### 原始代码（有bug）
```python
sampled_s.X = X
sampled_s = sampled_s.mask(node_mask, collapse=True)  # ❌ 破坏边信息
X, E, y = sampled_s.X, sampled_s.E, data.y
```

### 修复后（第1127-1159行）
```python
if enforce_scaffold:
    # 绕过mask，直接argmax
    X_indices = torch.argmax(X, dim=-1)  # [1, n, 8] → [1, n]
    E_indices = torch.argmax(E, dim=-1)  # [1, n, n, 5] → [1, n, n]
    
    # 验证边保留
    edge_check = []
    for i in range(min(10, E_indices.shape[1])):
        for j in range(i+1, min(10, E_indices.shape[2])):
            edge_type = E_indices[0, i, j].item()
            if edge_type < 4:
                edge_check.append(f"{i}-{j}:type{edge_type}")
    logging.info(f"  Edges preserved: {edge_check}")
    
    X = X_indices
    E = E_indices
    y = data.y
else:
    # 非骨架模式：使用原始mask
    sampled_s.X = X
    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, E, y = sampled_s.X, sampled_s.E, data.y
```

---

## ✅ 修复效果

### 预期日志

**Before (mask破坏)**:
```
First few edges: ['1-2:type1', ...]  ← 双键存在
Edges after mask: ['0-2:type0', ...]  ← 双键消失
Edge type counts: SINGLE=700, DOUBLE=0  ← 双键丢失
```

**After (绕过mask)**:
```
Edges preserved: ['1-2:type1', ...]  ← 双键保留
Edge type counts: SINGLE=35, DOUBLE=1, NO_EDGE=665  ← 正确！
Generated mol SMILES: CC(C)=CCCC...  ← 包含骨架！
✅ Generated molecule CONTAINS scaffold!
```

---

## 📊 技术细节

### argmax vs mask

| 操作 | 输入 | 输出 | 骨架保留 |
|------|------|------|----------|
| `mask(collapse=True)` | `[1, n, n, 5]` | `[1, n, n]` | ❌ 破坏 |
| `argmax(dim=-1)` | `[1, n, n, 5]` | `[1, n, n]` | ✅ 保留 |

**为什么argmax可以**：
- 直接取每个位置概率最高的类型
- 不改变索引映射（0-1还是0-1，1-2还是1-2）
- 保留了我们HOOK 3强制设置的概率（骨架位置概率=1）

**为什么mask会破坏**：
- 可能在内部做了节点重排
- 可能在聚合时用了错误的策略
- 设计给非骨架模式用的，不适合我们的场景

---

## 🚀 使用方法

```bash
modal run /Users/aylin/yaolab_projects/diffms_yaolab/modal/diffms_scaffold_inference.py
```

### 期望看到的新日志

```
[DEBUG] Bypassing mask to preserve scaffold edges
  X.shape before processing: torch.Size([1, 38, 8]), E.shape: torch.Size([1, 38, 38, 5])
  Edges preserved (first 10): ['0-1:type0', '1-2:type1', '2-3:type0', ...]
  
[DEBUG] Converting graph #0 to molecule:
  Edge type counts: SINGLE=35, DOUBLE=1, NO_EDGE=665
  
[DEBUG] Generated mol: CC(C)=CCCC(C(=O)O)C1CCC2...
  
✅ Generated molecule CONTAINS scaffold!
```

---

## 📝 相关文件

- **修改**: `/Users/aylin/yaolab_projects/diffms_yaolab/DiffMS/src/diffusion_model_spec2mol.py`
  - 第1127-1159行：新增绕过mask的逻辑

- **之前的调试**:
  - `docs/MASK_PROBLEM_20251028.md` - 问题诊断
  - `docs/CONVERSION_DEBUG_20251028.md` - 转换调试

---

## 💡 为什么这是正确的解决方案

1. **我们的X和E已经完美**: 
   - 初始化✓、HOOK 3冻结✓、最终验证✓
   
2. **mask是唯一的破坏点**:
   - mask前：边正确
   - mask后：边错乱
   
3. **argmax是安全的**:
   - 只做概率→索引的转换
   - 不改变图结构
   - 保留我们设置的所有约束

4. **向后兼容**:
   - 只在`enforce_scaffold=True`时绕过mask
   - 非骨架模式保持原样

---

**这是整个骨架约束功能的最后一块拼图！** 🎉

