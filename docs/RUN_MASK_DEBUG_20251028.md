# 🚀 运行mask调试版本

**目标**: 找出双键在哪里丢失

---

## ⚡ 立即运行

```bash
modal run /Users/aylin/yaolab_projects/diffms_yaolab/modal/diffms_scaffold_inference.py
```

---

## 📊 关键日志点

### 1️⃣ 初始化（应该有双键）

```
Bond 1-2: DOUBLE (idx=1)  ← 看到这个就OK
```

---

### 2️⃣ **新增：mask前检查**

```
[DEBUG] Before mask:
  E.shape = torch.Size([1, 38, 38, 5])  ← 应该是4D
  First few edges: ['0-1:type0', '1-2:type1', ...]  ← 看有没有type1
```

**关键问题**:
- ✅ `First few edges` 包含 `'1-2:type1'` → 双键在mask前存在
- ❌ 只有 `type0` → 双键在采样时就丢失了

---

### 3️⃣ **新增：mask后检查**

```
[DEBUG] After mask (collapse=True):
  E.shape = torch.Size([38, 38])  ← 变成2D
  E is 2D - checking first few values:
    E[0,1] = 0  ← SINGLE
    E[1,2] = ?  ← 关键！应该是1
```

**关键问题**:
- ✅ `E[1,2] = 1` → 双键保留了
- ❌ `E[1,2] = 0` → **mask丢失了双键！**

---

### 4️⃣ 转换前统计（修复了bug）

```
Edge type counts: SINGLE=?, DOUBLE=?, NO_EDGE=?
```

**期望**:
- SINGLE = 35~40
- DOUBLE = 1 (骨架的双键)
- NO_EDGE = 大量

**如果看到**:
- DOUBLE = 0 → 确认双键丢失

---

## 🎯 3种可能结果

### 结果 A: 采样时就丢了

```
[DEBUG] Before mask:
  First few edges: ['0-1:type0', '1-2:type0', ...]  ← 没有type1
```

→ 问题在HOOK 3，需要检查冻结逻辑

---

### 结果 B: mask时丢了（最可能）

```
[DEBUG] Before mask:
  First few edges: [..., '1-2:type1', ...]  ← 有type1

[DEBUG] After mask:
  E[1,2] = 0  ← 变成0了！
```

→ **mask(collapse=True) 有bug**，需要修复或绕过

---

### 结果 C: 统计bug（希望）

```
[DEBUG] After mask:
  E[1,2] = 1  ← 双键还在

Edge type counts: SINGLE=35, DOUBLE=1  ← 统计正确了
```

→ 之前只是统计代码的bug，已修复！

---

## 📝 下一步

运行后，复制以下关键日志：

```
1. Before mask:
   E.shape = ?
   First few edges: ?

2. After mask:
   E.shape = ?
   E[1,2] = ?

3. Edge type counts:
   SINGLE=?, DOUBLE=?
```

然后告诉我结果，我会立即提供针对性修复！🎯

