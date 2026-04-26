import torch
import torch.nn as nn
from Rvq import ResidualVectorQuantizer



class EncoderLayerBN(nn.Module):
    """
    Couche d'encodage standard intégrant des convolutions, une normalisation spatiale, 
    et optionnellement un sous-échantillonnage par regroupement maximum.
    """
    def __init__(self, ch_in, ch_out, kernel_size, padding, pooling, dropout):
        super(EncoderLayerBN, self).__init__()

        self.pooling = nn.MaxPool2d(2) if pooling else None

        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        return self.block(x)


class DecoderLayerBN(nn.Module):
    """
    Couche de décodage gérant le suréchantillonnage et la fusion des caractéristiques résiduelles.
    """
    def __init__(self, ch_in, ch_out, kernel_size, padding, dropout, 
                 skip_mode= "none", upsampling_mode="transpose", cropping=False):
        super(DecoderLayerBN, self).__init__()

        self.cropping = cropping
        self.skip_mode = skip_mode
        self.upsampling_mode = upsampling_mode

        if self.upsampling_mode == "transpose":
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

        if self.skip_mode == "concat":
            ch_hidden = ch_out * 2
        else:
            ch_hidden = ch_out

        self.block = nn.Sequential(
            nn.Conv2d(ch_hidden, ch_out, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def crop(self, x, cropping_size):
        """
        Rogne symétriquement un tenseur spatialement pour correspondre à une dimension cible.
        """
        h_crop, w_crop = cropping_size[0].item(), cropping_size[1].item()
        if h_crop == 0 and w_crop == 0:
            return x

        h_end = -h_crop if h_crop > 0 else None
        w_end = -w_crop if w_crop > 0 else None

        return x[:, :, h_crop:h_end, w_crop:w_end]

    def forward(self, x, skip_features):
        if self.upsampling_mode == "transpose":
            x = self.up(x)
        else:
            x = self.conv(self.up(x))

        if self.cropping and skip_features is not None:
            cropping_size = (torch.tensor(skip_features.shape[2:]) - torch.tensor(x.shape[2:])) // 2
            skip_features = self.crop(skip_features, cropping_size)

        if self.skip_mode == "concat" and skip_features is not None:
            x = self.block(torch.cat((x, skip_features), 1))
        elif self.skip_mode == "add" and skip_features is not None:
            x = self.block(x + skip_features)
        elif self.skip_mode == "none":
            x = self.block(x)

        return x


class UNet2d(nn.Module):
    """
    Architecture U-Net configurable intégrant optionnellement une quantification vectorielle résiduelle.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 encoder_layer=EncoderLayerBN,
                 decoder_layer=DecoderLayerBN,
                 hidden_dims=[64, 128, 256, 512, 1024],
                 kernel_size=3,
                 padding_mode="valid",
                 skip_mode="none",
                 upsampling_mode="transpose",
                 dropout=0.0,
                 use_rvq=False,
                 num_quantizers=4,
                 codebook_size=1024):
        
        super(UNet2d, self).__init__()

        assert len(hidden_dims) > 0, "UNet2d nécessite au moins une dimension cachée."
        assert padding_mode in ["same", "valid"], "Le mode de padding doit être 'same' ou 'valid'."

        self.padding_mode = padding_mode
        self.use_rvq = use_rvq

        cropping = (padding_mode == "valid")
        padding = 0 if padding_mode == "valid" else kernel_size // 2

        # Construction de l'encodeur
        encoder = []
        for i in range(len(hidden_dims)):
            ch_in = input_dim if i == 0 else hidden_dims[i-1]
            ch_out = hidden_dims[i]
            is_last = (i == len(hidden_dims) - 1)
            encoder.append(encoder_layer(
                ch_in, ch_out, kernel_size=kernel_size, padding=padding, 
                pooling=(i > 0), dropout=dropout if is_last else 0.0
            ))
        self.encoder = nn.ModuleList(encoder)

        # Initialisation conditionnelle du RVQ
        if self.use_rvq:
            latent_dim = hidden_dims[-1]
            self.rvq = ResidualVectorQuantizer(
                num_quantizers=num_quantizers,
                num_embeddings=codebook_size,
                embedding_dim=latent_dim
            )

        # Construction du décodeur
        decoder = []
        hidden_dims_rev = hidden_dims[::-1]

        for i in range(len(hidden_dims_rev) - 1):
            ch_in = hidden_dims_rev[i]
            ch_out = hidden_dims_rev[i+1]
            decoder.append(decoder_layer(
                ch_in, ch_out, kernel_size=kernel_size, padding=padding, dropout=0.0, 
                skip_mode=skip_mode, upsampling_mode=upsampling_mode, cropping=cropping
            ))
        self.decoder = nn.ModuleList(decoder)

        self.final_conv = nn.Conv2d(hidden_dims[0], output_dim, kernel_size=1, stride=1, padding=0)
        self.final_act = nn.Tanh()

    def encode(self, x):
        """
        Extrait la représentation latente et les caractéristiques spatiales intermédiaires.
        """
        skip_features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_features.insert(0, x)

        bottleneck = skip_features[0]
        skips = skip_features[1:]
        return bottleneck, skips

    def decode(self, z, skip_features=None):
        """
        Génère une reconstruction spatiale à partir d'un tenseur latent.
        """
        x = z
        for i, decoder_layer in enumerate(self.decoder):
            skip = skip_features[i] if skip_features is not None else None
            x = decoder_layer(x, skip)

        x = self.final_conv(x)
        return self.final_act(x)

    def forward(self, x):
        """
        Effectue une passe avant complète avec quantification optionnelle.
        """
        z, skips = self.encode(x)
        
        rvq_loss = torch.tensor(0.0, device=x.device)
        rvq_info = {}
        
        if self.use_rvq:
            z, rvq_loss, all_indices, rvq_metrics = self.rvq(z)
            rvq_info['indices'] = all_indices
            rvq_info['metrics'] = rvq_metrics
            
        reconstructed = self.decode(z, skips)
        
        return reconstructed, rvq_loss, rvq_info


def weights_init(m):
    """
    Initialise les poids du réseau de neurones selon des heuristiques standards.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.2)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
