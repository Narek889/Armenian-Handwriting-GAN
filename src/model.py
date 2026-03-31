import torch
import torch.nn as nn


# ── Helpers ───────────────────────────────────────────────────────────────────

def weights_init(m):
    # Կանչվում է self.apply(weights_init)-ով — բոլոր շերտերի վրա
    # Conv շերտերի weight-երը նախաձևավորում ենք N(0, 0.02) բաշխումով
    # Փոքր std=0.02 → սկզբում weight-երը մոտ են 0-ի, GAN-ը կայուն է սկսում
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Input:  noise z (latent_dim,)  +  class label → embedding
    Output: 1 × 64 × 64 image, Tanh [-1, 1]

    """

    def __init__(
        self,
        num_classes: int = 78,
        latent_dim: int = 100,
        embed_dim: int = 100,
        ngf: int = 64,
        img_channels: int = 1,        # grayscale = 1, RGB = 3
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim  = embed_dim
        self.ngf        = ngf

        # label_emb — դասի համարը (0-77) վերածում է վեկտորի
        # Օրինակ՝ class 19 ('Մ') → [0.3, -0.7, 0.1, ...] 100 թիվ
        # Մոդելը սովորում է յուրաքանչյուր տառի "իմաստը" այս embedding-ում
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        # noise[100] + embedding[100] = [200] → մեկ հարթ վեկտոր
        input_dim = latent_dim + embed_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, ngf * 8 * 4 * 4),   # 200 → 131072 (512×4×4)
            nn.ReLU(inplace=True),
        )

        self.conv_blocks = nn.Sequential(
            # 4×4 → 8×8
            # ConvTranspose2d — հակառակ Conv, փոքր feature map-ը մեծացնում է
            # SpectralNorm — weight matrix-ի մեծագույն singular value-ն ≤ 1
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 8×8 → 16×16
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 16×16 → 32×32
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 32×32 → 64×64
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False)),
            nn.Tanh(),               # Ելքը [-1,1] — generate.py-ում (img+1)/2 → [0,1]
        )

        self.apply(weights_init)    # բոլոր շերտերին կիրառում ենք նախաձևավորումը

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels)                    # label → embedding վեկտոր
        x   = torch.cat([noise, emb], dim=1)    # noise + label = [200] մուտք
        x   = self.fc(x)                                # [200] → [131072]
        x   = x.view(x.size(0), self.ngf * 8, 4, 4)     # reshape → (B, 512, 4, 4)
        return self.conv_blocks(x)


# ── Discriminator (Critic) ────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Input:  1 × 64 × 64 image
    Output:
        validity  — raw scalar score (WGAN critic) OR probability (BCE mode)
        class_out — (num_classes,) logits for auxiliary classification
    """

    def __init__(
        self,
        num_classes: int = 78,
        ndf: int = 64,
        img_channels: int = 1,
        wgan: bool = True,   # True = raw score (WGAN), False = Sigmoid (BCE)
    ):
        super().__init__()
        self.ndf  = ndf
        self.wgan = wgan
        # No BatchNorm — use InstanceNorm instead (safe with GP)
        # InstanceNorm — ամեն նկար ինքն իր նկատmամբ է նormalացվum
        # BN-ի փոխարեն, batch-ից կaxված չէ → GP-ն ճիշտ է աշխատում
        self.features = nn.Sequential(
            # 64×64 → 32×32
            # Առաջին շերտում InstanceNorm չկա — դեռ բավական տեղեկատվություն կա
            nn.utils.spectral_norm(nn.Conv2d(img_channels, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # 32×32 → 16×16
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 16×16 → 8×8
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 8×8 → 4×4
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )  # output: (B, ndf*8, 4, 4)

        flat_dim = ndf * 8 * 4 * 4   # 64*8*4*4 = 8192 — հարթ վեկտոր

        # Adversarial head — "որqա՞ն իդեալական է"
        # WGAN mode  → raw թիվ, կարող է լինեկ բացասական (-5.2) կամ դրական (+3.7)
        # BCE mode   → Sigmoid → [0,1] հավանականություն
        if wgan:
            # Raw score — no Sigmoid
            self.adv_head = nn.utils.spectral_norm(nn.Linear(flat_dim, 1))
        else:
            self.adv_head = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(flat_dim, 1)),
                nn.Sigmoid(),
            )

        # cls_head — auxiliary classifier, ո՞ր դասն է (78 Armenian characters)
        # Սա AC-GAN-ի «AC» մասն է (Auxiliary Classifier)
        self.cls_head = nn.utils.spectral_norm(nn.Linear(flat_dim, num_classes))

        self.apply(weights_init)

    def forward(self, img: torch.Tensor):
        feat      = self.features(img)
        feat_flat = feat.view(feat.size(0), -1)
        validity  = self.adv_head(feat_flat)
        class_out = self.cls_head(feat_flat)
        return validity, class_out