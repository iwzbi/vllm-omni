import torch
import torch.nn.functional as F

# 设置随机种子以便复现
torch.manual_seed(42)

# 假设序列长度为3，嵌入维度为4
seq_len = 3
d_model = 4

# 随机生成输入（代表3个token的嵌入）
x = torch.randn(seq_len, d_model)
print("Input tokens (x):")
print(x)

# 线性变换得到 Q, K, V（模拟self-attention中的投影）
W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)

Q = x @ W_q  # [3, 4]
K = x @ W_k  # [3, 4]
V = x @ W_v  # [3, 4]
print(K)

print("\nOriginal Q, K, V (correct alignment):")
print("Q[0] corresponds to token0, K[0]/V[0] also token0, etc.")

# === 正确的 attention（causal=False）===
attn_scores = Q @ K.T  # [3, 3]
attn_weights = F.softmax(attn_scores, dim=-1)  # [3, 3]
output_correct = attn_weights @ V  # [3, 4]

print("\nCorrect output:")
print(output_correct)

# === 打乱 K 和 V 的顺序 ===
# 原顺序: [0, 1, 2] → 打乱为 [1, 2, 0]
perm = [1, 2, 0]
K_shuffled = K[perm]  # K[1], K[2], K[0]
V_shuffled = V[perm]  # V[1], V[2], V[0]
print(K_shuffled)

print("\nShuffled K and V order: [1, 2, 0]")
print("Now Q[0] (token0) will attend to K[1]/V[1] (which is token1's info) as 'position 0'")

# 计算打乱后的 attention
attn_scores_shuffled = Q @ K_shuffled.T  # [3, 3]
attn_weights_shuffled = F.softmax(attn_scores_shuffled, dim=-1)
output_shuffled = attn_weights_shuffled @ V_shuffled  # [3, 4]

print("\nShuffled output:")
print(output_shuffled)

# === 对比结果 ===
print("\nAre the outputs equal?")
print(torch.allclose(output_correct, output_shuffled, atol=1e-6))  # 应该是 False

print("\nDifference:")
print(output_correct - output_shuffled)

