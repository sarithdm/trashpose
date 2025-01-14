import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_rgb, dim_flow, dim_pose, hidden_dim):
        super(CrossAttentionFusion, self).__init__()
        self.rgb_encoder = nn.Linear(dim_rgb, hidden_dim)
        self.flow_encoder = nn.Linear(dim_flow, hidden_dim)
        self.pose_encoder = nn.Linear(dim_pose, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, rgb_features, flow_features, pose_features):
        # Encode features
        rgb_encoded = self.rgb_encoder(rgb_features)
        flow_encoded = self.flow_encoder(flow_features)
        pose_encoded = self.pose_encoder(pose_features)

        # Cross-attention (query: rgb, keys & values: flow + pose)
        combined_features = torch.cat([flow_encoded.unsqueeze(0), pose_encoded.unsqueeze(0)], dim=0)
        attn_output, _ = self.cross_attention(rgb_encoded.unsqueeze(0), combined_features, combined_features)

        # Final classification
        output = self.fc(attn_output.squeeze(0))
        return output
