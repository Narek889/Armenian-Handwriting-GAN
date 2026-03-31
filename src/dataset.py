"""
dataset.py  —  Mashtots dataset loader for AC-GAN
Folder structure expected:
    data/
      0/  (Ա)  img1.png, img2.png ...
      1/  (Բ)  ...
      ...
      77/ (ֆ)  ...
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ── Armenian character map: class_id → character ──────────────────────────────
CLASS_TO_CHAR = {
    0:'Ա', 1:'Բ', 2:'Գ', 3:'Դ', 4:'Ե', 5:'Զ', 6:'Է', 7:'Ը', 8:'Թ', 9:'Ժ',
    10:'Ի', 11:'Լ', 12:'Խ', 13:'Ծ', 14:'Կ', 15:'Հ', 16:'Ձ', 17:'Ղ', 18:'Ճ',
    19:'Մ', 20:'Յ', 21:'Ն', 22:'Շ', 23:'Ո', 24:'Ու', 25:'Չ', 26:'Պ', 27:'Ջ',
    28:'Ռ', 29:'Ս', 30:'Վ', 31:'Տ', 32:'Ր', 33:'Ց', 34:'Փ', 35:'Ք', 36:'Եվ',
    37:'Օ', 38:'Ֆ',
    39:'ա', 40:'բ', 41:'գ', 42:'դ', 43:'ե', 44:'զ', 45:'է', 46:'ը', 47:'թ',
    48:'ժ', 49:'ի', 50:'լ', 51:'խ', 52:'ծ', 53:'կ', 54:'հ', 55:'ձ', 56:'ղ',
    57:'ճ', 58:'մ', 59:'յ', 60:'ն', 61:'շ', 62:'ո', 63:'ու', 64:'չ', 65:'պ',
    66:'ջ', 67:'ռ', 68:'ս', 69:'վ', 70:'տ', 71:'ր', 72:'ց', 73:'փ', 74:'ք',
    75:'և', 76:'օ', 77:'ֆ',
}

# Հակառակ արտապատկերում — տառ → դասի համար
# Օգտագործվում է կանխատեսումը տառի վերածելու համար
CHAR_TO_CLASS = {v: k for k, v in CLASS_TO_CHAR.items()}
NUM_CLASSES = len(CLASS_TO_CHAR)  # 78


def get_transforms(img_size: int = 64) -> T.Compose:
    """
    Նկարները մոդելի համար պատրաստող pipeline.
    1. Grayscale  — գունավոր → սևուսպիտակ (տառերի համար գույնը կարևոր չէ)
    2. Resize     — բոլոր նկարները հավասարեցնել 64×64 չափի
    3. ToTensor   — PIL Image → PyTorch tensor [0,1] արժեքներով
    4. Normalize  — [0,1] → [-1,1] (ավելի կայուն սովորեցման համար)
    """

    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((img_size, img_size)),
        T.ToTensor(),                          # [0, 1]
        T.Normalize([0.5], [0.5]),             # [-1, 1]
    ])


class MashtotsDataset(Dataset):
    """
    Loads Mashtots from folder-per-class structure.
    Each folder name is the integer class id (0..77).
    """

    SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

    def __init__(self, root: str, img_size: int = 64, transform=None):
        self.root = root
        self.transform = transform or get_transforms(img_size)
        self.samples: list[tuple[str, int]] = []  # (path, class_id)
        self._load_samples()

    def _load_samples(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        found_classes = set()
        # Թղթապանակները դասավորել ըստ թվային կարգի
        for folder in sorted(os.listdir(self.root), key=lambda x: int(x) if x.isdigit() else -1):
            folder_path = os.path.join(self.root, folder)
            if not os.path.isdir(folder_path):
                continue
            if not folder.isdigit():
                continue
            class_id = int(folder)
            if class_id not in CLASS_TO_CHAR:
                print(f"[WARNING] Unknown class folder: {folder}, skipping.")
                continue
            found_classes.add(class_id)
            for fname in os.listdir(folder_path):
                if os.path.splitext(fname)[1].lower() in self.SUPPORTED_EXTS:
                    self.samples.append((os.path.join(folder_path, fname), class_id))

        print(f"[Dataset] Loaded {len(self.samples)} images from {len(found_classes)} classes.")

        # Ստուգել արդյոք բոլոր 78 դասերն ունեն նկարներ
        if len(found_classes) < NUM_CLASSES:
            missing = set(range(NUM_CLASSES)) - found_classes
            print(f"[Dataset] Missing classes: {sorted(missing)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_id = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(class_id, dtype=torch.long)


def get_dataloader(
    root: str,
    batch_size: int = 64,
    img_size: int = 64,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    dataset = MashtotsDataset(root=root, img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
