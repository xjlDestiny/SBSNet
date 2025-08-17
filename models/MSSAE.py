

import torch
from torch import nn
from torchinfo import summary

from models.transformerEncoder import TransformerEncoder


def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    '''
    returns a block conv-bn-relu
    '''
    layers = [
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    ]
    return nn.Sequential(*layers)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class AttentionBlock(nn.Module):
    def __init__(self, channel, reduction=2, ffn_hidden=128, n_head=2, n_layers=1,
                 max_len=256, drop_prob=0.1, device='cpu'):
        super(AttentionBlock, self).__init__()
        self.channelAttention = SEBlock(channel, reduction)
        self.specAttention = TransformerEncoder(channel,ffn_hidden, n_head,
                                                n_layers, max_len, drop_prob, device)

    def forward(self, x):
        x_c, x_s = x.chunk(2, dim=1)
        x_s = x_s.permute(0, 2, 1)

        out_c = self.channelAttention(x_c)
        out_s = self.specAttention(x_s)
        out_s = out_s.permute(0, 2, 1)
        out = torch.cat([out_c, out_s], dim=1)

        return out

class EncoderBlock(nn.Module):
     def __init__(self, in_len, mapped_len, d_input=1, d_model=32, layers=2, ffn_hidden=128,
                 n_head=2, n_layers=1, max_len=256, drop_prob=0.1, device='cpu'):
         super(EncoderBlock, self).__init__()

         self.in_map = nn.Sequential(
             nn.Linear(in_len, mapped_len),
             nn.Conv1d(d_input, d_model, kernel_size=3, padding=1)
         )
         self.layer1 = nn.Sequential(*[nn.Sequential(
             conv1d_block(d_model, d_model, kernel_size=7, stride=2, padding=3),
             AttentionBlock(d_model // 2, 2, ffn_hidden, n_head, n_layers, max_len, drop_prob, device))
             for _ in range(layers)
         ])
         self.layer2 = nn.Sequential(*[nn.Sequential(
             conv1d_block(d_model, d_model, kernel_size=5, stride=2, padding=2),
             AttentionBlock(d_model // 2, 2, ffn_hidden, n_head, n_layers, max_len, drop_prob, device))
             for _ in range(layers)
         ])
         self.layer3 = nn.Sequential(*[nn.Sequential(
             conv1d_block(d_model, d_model, kernel_size=3, stride=1, padding=1),
             nn.MaxPool1d(2),
             AttentionBlock(d_model // 2, 2, ffn_hidden, n_head, n_layers, max_len, drop_prob, device))
             for _ in range(layers)
         ])


         self.out_map = nn.Conv1d(d_model, pow(2, layers), kernel_size=3, padding=1)

     def forward(self, x):
         # x.shape ==> (batch_size, C, H)
         x = x.unsqueeze(-2)
         mapped_x = self.in_map(x)

         out_layer1 = self.layer1(mapped_x)
         out_layer2 = self.layer2(mapped_x)
         out_layer3 = self.layer3(mapped_x)
         middle_out = (out_layer1 + out_layer2 + out_layer3)

         encoded_out = self.out_map(middle_out)
         encoded_out = encoded_out.reshape(encoded_out.shape[0], -1)

         return middle_out, encoded_out

class DecoderBlock(nn.Module):
    def __init__(self, in_len, out_len, d_input, d_output=1, use_unpooling=False, layers=2, ffn_hidden=128,
                 n_head=2, n_layers=1, max_len=256, drop_prob=0.1, device='cpu'):
        super(DecoderBlock, self).__init__()
        self.use_unpooling = use_unpooling

        if self.use_unpooling:
            # Use Unpooling ==> (output_shape - 1) * stride - 2 * padding + kernel_size + output_padding
            self.decoder = nn.Sequential(*[nn.Sequential(
                nn.ConvTranspose1d(d_input, d_input, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(d_input),
                nn.ReLU(),
                AttentionBlock(d_input // 2, 2, ffn_hidden, n_head, n_layers, max_len, drop_prob, device))
                for _ in range(layers)
            ])
        else:
            # Use Upsampling
            self.decoder = nn.Sequential(*[nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv1d_block(d_input, d_input, kernel_size=3, stride=1, padding=1),
                AttentionBlock(d_input // 2, 2, ffn_hidden, n_head, n_layers, max_len, drop_prob, device))
                for _ in range(layers)
            ])

        self.recon_map = nn.Sequential(
            nn.Conv1d(d_input, d_output, kernel_size=3, padding=1),
            nn.Linear(in_len, out_len)
        )

    def forward(self, x):
        out = self.decoder(x)
        decoded_out = self.recon_map(out)
        decoded_out = decoded_out.reshape(decoded_out.shape[0], -1)
        return decoded_out


class MSSAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(MSSAE, self).__init__()
        # b, c, h, w
        # self.x = data
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_activations = None

    def forward(self, x):
        """

        Args:
            x: Batch spectral data ——> (N, B), 其中N为批量大小，B为波段数

        Returns:
            编码器输出， 解码器输出
        """
        self.hidden_activations, encoded_out = self.encoder(x)
        decoded_out = self.decoder(self.hidden_activations)
        return encoded_out, decoded_out

if __name__ == "__main__":

    device = torch.device('cpu')

    # # test SEBlock
    # se_model = SEBlock(16, 2)
    # x = torch.randn((32, 16, 32))
    # y = se_model(x)
    # print(y.shape)

    # test EncoderBlock
    encoder = EncoderBlock(205, 128).to(device)
    x1 = torch.randn((64, 205)).to(device)
    summary(encoder, (64, 205), device=device)
    _, y1 = encoder(x1)
    print(y1.shape)

    # test DecoderBlock
    decoder = DecoderBlock(128, 205, 32, use_unpooling=False).to(device)
    x_d = torch.randn((64, 32, 32)).to(device)
    y_d = decoder(x_d)
    print(x_d.shape)
    print(y_d.shape)

    mssae_model = MSSAE(encoder, decoder).to(device)
    summary(mssae_model, (64, 205), device=device)
    x_mssae = torch.randn((64, 205)).to(device)
    encoded_y, decoded_y = mssae_model(x_mssae)
    print(x_mssae.shape)
    print(encoded_y.shape)
    print(decoded_y.shape)