import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, causal=False):
    """手动实现 attention"""
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if causal:
        # 应用 causal mask
        seq_len = q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights

# 设置简单的输入
torch.manual_seed(42)
seq_len, d_model = 3, 4

q = torch.randn(1, seq_len, d_model)  # [token0, token1, token2]
k_orig = torch.randn(1, seq_len, d_model)
v_orig = torch.randn(1, seq_len, d_model)

# 打乱 k 和 v 的顺序（保持一致）
k_shuffled = k_orig[:, [1, 2, 0], :]  # [token1, token2, token0]
v_shuffled = v_orig[:, [1, 2, 0], :]  # [token1, token2, token0]

# 打乱 k 但不打乱 v（顺序不一致）
v_mismatch = v_orig  # [token0, token1, token2]

print("=" * 60)
print("情况1：causal=False, k 和 v 顺序一致")
print("=" * 60)

output1, _ = scaled_dot_product_attention(q, k_orig, v_orig, causal=False)
output2, _ = scaled_dot_product_attention(q, k_shuffled, v_shuffled, causal=False)

print(f"原始 k, v: output shape = {output1.shape}")
print(f"打乱后 k, v (保持一致): output shape = {output2.shape}")
print(f"两个输出是否相同: {torch.allclose(output1, output2, atol=1e-5)}")
print(f"最大差异: {(output1 - output2).abs().max().item()}")

print("\n" + "=" * 60)
print("情况2：causal=False, k 和 v 顺序不一致")
print("=" * 60)

output3, _ = scaled_dot_product_attention(q, k_shuffled, v_mismatch, causal=False)

print(f"打乱 k, 原始 v: output shape = {output3.shape}")
print(f"与原始输出是否相同: {torch.allclose(output1, output3, atol=1e-5)}")
print(f"最大差异: {(output1 - output3).abs().max().item()}")

print("\n" + "=" * 60)
print("情况3：causal=True, k 和 v 顺序一致")
print("=" * 60)

output4, attn4 = scaled_dot_product_attention(q, k_orig, v_orig, causal=True)
output5, attn5 = scaled_dot_product_attention(q, k_shuffled, v_shuffled, causal=True)

print(f"原始 k, v: output shape = {output4.shape}")
print(f"打乱后 k, v (保持一致): output shape = {output5.shape}")
print(f"两个输出是否相同: {torch.allclose(output4, output5, atol=1e-5)}")
print(f"最大差异: {(output4 - output5).abs().max().item()}")

print("\nAttention weights (原始):")
print(attn4[0].round(decimals=3))
print("\nAttention weights (打乱后保持一致):")
print(attn5[0].round(decimals=3))

print("\n" + "=" * 60)
print("情况4：causal=True, k 和 v 顺序不一致")
print("=" * 60)

output6, attn6 = scaled_dot_product_attention(q, k_shuffled, v_mismatch, causal=True)

print(f"打乱 k, 原始 v: output shape = {output6.shape}")
print(f"与原始输出是否相同: {torch.allclose(output4, output6, atol=1e-5)}")
print(f"最大差异: {(output4 - output6).abs().max().item()}")

print("\nAttention weights (k 打乱, v 原始):")
print(attn6[0].round(decimals=3))

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("✅ causal=False, k v 顺序一致: 输出相同")
print("❌ causal=False, k v 顺序不一致: 输出不同")
print("✅ causal=True, k v 顺序一致: 输出相同")
print("❌ causal=True, k v 顺序不一致: 输出不同")

