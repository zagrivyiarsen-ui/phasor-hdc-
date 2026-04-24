
# -*- coding: utf-8 -*-
"""
Experiment 1: Colored MNIST – Causal Discovery and Do‑Intervention
==================================================================
FINAL VERSION – READY FOR PUBLICATION

Reproduces:
- Unsupervised slot learning (Barlow Twins + orthogonality)
- Supervised slot learning (with ground truth labels for object and background)
- Causal logic training with knockoff (invariance to background)
- Algebraic do‑intervention (forcing attention to object only)
- Comprehensive analysis: attention dynamics, t‑SNE, slot cosine similarity

All results are saved in ./results/exp1/
Figures are saved as PDF (vector) and PNG.
"""

import os
import json
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ===============================
# CONFIGURATION – all hyperparameters in one place
# ===============================
CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_seeds': 10,                     # statistical significance
    'batch_size': 64,
    'd_slot': 2000,                      # hyperdimensionality for slots
    'train_samples': 5000,
    'test_samples': 1000,
    'corr_train': 0.95,
    'corr_test_iid': 0.95,
    'corr_test_ood': 0.95,
    'invert_ood': True,
    'epochs_slot_unsup': 150,
    'epochs_slot_sup': 30,
    'epochs_logic': 100,
    'lr_slot': 1e-3,
    'lr_logic': 0.1,
    'lambda_orth': 0.1,
    'num_workers': 2,
    'results_dir': './results/exp1'
}
os.makedirs(CONFIG['results_dir'], exist_ok=True)
os.makedirs(os.path.join(CONFIG['results_dir'], 'plots'), exist_ok=True)
os.makedirs(os.path.join(CONFIG['results_dir'], 'models'), exist_ok=True)
device = CONFIG['device']
print(f"Device: {device}")
print(f"Results will be saved in: {CONFIG['results_dir']}\n")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===============================
# DATASET: Colored MNIST
# ===============================
class ColoredMNIST(Dataset):
    def __init__(self, num_samples, corr=0.95, invert=False, img_size=28):
        self.num_samples = num_samples
        self.corr = corr
        self.invert = invert
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.colors = []
        self._generate()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _generate(self):
        for _ in range(self.num_samples):
            label = torch.randint(0, 2, (1,)).item()
            if not self.invert:
                color = label if np.random.rand() < self.corr else (1 - label)
            else:
                color = 1 - label

            img = torch.zeros(3, self.img_size, self.img_size)
            if color == 0:
                img[0, :, :] = 0.8   # red background
            else:
                img[2, :, :] = 0.8   # blue background

            if label == 0:  # circle
                y, x = torch.meshgrid(torch.arange(self.img_size),
                                      torch.arange(self.img_size), indexing='ij')
                mask = (x - self.img_size//2)**2 + (y - self.img_size//2)**2 <= (self.img_size//3)**2
                img[:, mask] = 1.0
            else:           # square
                s = self.img_size // 3
                img[:, s:self.img_size-s, s:self.img_size-s] = 1.0

            self.images.append(img)
            self.labels.append(label)
            self.colors.append(color)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.normalize(self.images[idx])
        return img, self.labels[idx], self.colors[idx]

# ===============================
# FEATURE EXTRACTOR (frozen ResNet-18)
# ===============================
def get_feature_extractor():
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone.to(device)

@torch.no_grad()
def extract_features(loader, backbone):
    all_feat, all_labels, all_colors = [], [], []
    for imgs, lbl, col in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device)
        feat = backbone(imgs)
        feat = feat.view(feat.size(0), -1)
        all_feat.append(feat.cpu())
        all_labels.append(lbl)
        all_colors.append(col)
    return torch.cat(all_feat), torch.cat(all_labels), torch.cat(all_colors)

# ===============================
# HYPERDIMENSIONAL PRIMITIVES
# ===============================
def generate_phasor(dim, num_vectors=1):
    theta = (torch.rand(num_vectors, dim) * 2 * np.pi) - np.pi
    return torch.polar(torch.ones_like(theta), theta)

def fast_cosine_sim(A, B):
    return torch.real(torch.sum(A * torch.conj(B), dim=-1)) / A.shape[-1]

# ===============================
# SLOT MODEL (shared for unsup/sup)
# ===============================
class SlotModel(nn.Module):
    def __init__(self, feat_dim=512, D=CONFIG['d_slot'], num_slots=2):
        super().__init__()
        self.num_slots = num_slots
        self.D = D
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_slots * 2)
        )
        # Basis shape: [D, num_slots*2]
        basis = generate_phasor(D, num_slots * 2).T  # [D, num_slots*2]
        self.basis = nn.Parameter(basis, requires_grad=True)

    def forward(self, x):
        logits = self.proj(x)
        logits = logits.view(logits.size(0), self.num_slots, 2)
        probs = F.softmax(logits, dim=-1)   # [B, num_slots, 2]
        Z = []
        for i in range(self.num_slots):
            Z_i = (probs[:, i, 0:1] * self.basis[:, 2*i] +
                   probs[:, i, 1:2] * self.basis[:, 2*i+1])
            Z.append(Z_i)
        return Z, probs

# ===============================
# CAUSAL LOGIC MODULE
# ===============================
class CausalLogic(nn.Module):
    def __init__(self, D=CONFIG['d_slot']):
        super().__init__()
        self.attn_logits = nn.Parameter(torch.tensor([0.0, 0.0], device=device))
        rule_phasor = generate_phasor(D, 1).squeeze(0)
        self.rule = nn.Parameter(rule_phasor)
        action_a = generate_phasor(D, 1).squeeze(0)
        self.register_buffer('action_A', action_a)
        self.register_buffer('action_B', -action_a)

    def forward(self, z_obj, z_bg, tau=0.5):
        batch_size = z_obj.size(0)
        attn_logits_exp = self.attn_logits.unsqueeze(0).expand(batch_size, -1)
        attn = F.gumbel_softmax(attn_logits_exp, tau=tau, hard=False, dim=-1)
        w_obj = attn[:, 0].unsqueeze(-1)
        w_bg = attn[:, 1].unsqueeze(-1)
        Z_cause = w_obj * z_obj + w_bg * z_bg
        Z_pred = Z_cause * self.rule
        sim_a = fast_cosine_sim(Z_pred, self.action_A)
        sim_b = fast_cosine_sim(Z_pred, self.action_B)
        return sim_a - sim_b

# ===============================
# TRAINING HELPERS
# ===============================
def augment_features(feat):
    """Feature‑level augmentation for Barlow Twins."""
    feat1 = feat + torch.randn_like(feat) * 0.1
    feat2 = feat + torch.randn_like(feat) * 0.1
    mask1 = torch.rand_like(feat) > 0.2
    mask2 = torch.rand_like(feat) > 0.2
    feat1 = feat1 * mask1
    feat2 = feat2 * mask2
    return feat1, feat2

def barlow_twins_loss(z1, z2, lambda_off=0.005):
    ang1 = torch.angle(z1)
    ang2 = torch.angle(z2)
    ang1 = ang1 - ang1.mean(dim=0, keepdim=True)
    ang2 = ang2 - ang2.mean(dim=0, keepdim=True)
    B, C = ang1.shape
    corr = (ang1.T @ ang2) / (B - 1)
    diag = torch.diag(corr)
    off_diag = corr - torch.diag(diag)
    loss = (diag - 1).pow(2).sum() + lambda_off * off_diag.pow(2).sum()
    return loss

def train_unsupervised_slots(slot_model, train_feat, epochs, device, seed):
    seed_everything(seed)
    opt = optim.Adam(slot_model.parameters(), lr=CONFIG['lr_slot'])
    for epoch in range(epochs):
        total_loss = 0
        idx = torch.randperm(train_feat.size(0))[:256]
        num_batches = 0
        for i in range(0, len(idx), CONFIG['batch_size']):
            batch_idx = idx[i:i+CONFIG['batch_size']]
            feat_batch = train_feat[batch_idx].to(device)
            f1, f2 = augment_features(feat_batch)
            Z1, _ = slot_model(f1)
            Z2, _ = slot_model(f2)
            loss_bt = barlow_twins_loss(Z1[0], Z2[0]) + barlow_twins_loss(Z1[1], Z2[1])
            # Orthogonality regularization between the two slots
            sim_slots = fast_cosine_sim(Z1[0], Z1[1]).mean()
            loss_orth = torch.abs(sim_slots)
            loss = loss_bt + CONFIG['lambda_orth'] * loss_orth
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            num_batches += 1
        if (epoch+1) % 30 == 0:
            avg_loss = total_loss / num_batches
            print(f"      Unsup slot epoch {epoch+1}: loss={avg_loss:.2f}")

def train_supervised_slots(slot_model, train_feat, train_labels, train_colors, device, seed):
    seed_everything(seed)
    opt = optim.Adam(slot_model.parameters(), lr=CONFIG['lr_slot'])
    for epoch in range(CONFIG['epochs_slot_sup']):
        total_loss = 0
        idx = torch.randperm(train_feat.size(0))[:256]
        num_batches = 0
        for i in range(0, len(idx), CONFIG['batch_size']):
            batch_idx = idx[i:i+CONFIG['batch_size']]
            feat_batch = train_feat[batch_idx].to(device)
            lbl_batch = train_labels[batch_idx].to(device)
            col_batch = train_colors[batch_idx].to(device)
            _, probs = slot_model(feat_batch)
            loss_obj = F.cross_entropy(probs[:, 0, :], lbl_batch)
            loss_bg = F.cross_entropy(probs[:, 1, :], col_batch)
            Z_list, _ = slot_model(feat_batch)
            orth = torch.abs(fast_cosine_sim(Z_list[0], Z_list[1])).mean()
            loss = loss_obj + loss_bg + CONFIG['lambda_orth'] * orth
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            num_batches += 1
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"      Sup slot epoch {epoch+1}: loss={avg_loss:.4f}")

def train_causal_logic(logic, Z_obj, Z_bg, train_labels, device, seed):
    seed_everything(seed)
    opt = optim.Adam(logic.parameters(), lr=CONFIG['lr_logic'])
    scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=30)
    for epoch in range(CONFIG['epochs_logic']):
        idx_shuffle = torch.randperm(Z_bg.size(0), device=device)
        Z_bg_shuffled = Z_bg[idx_shuffle]
        logits_real = logic(Z_obj, Z_bg)
        loss_real = F.binary_cross_entropy_with_logits(logits_real, train_labels.float())
        logits_fake = logic(Z_obj, Z_bg_shuffled)
        loss_fake = F.binary_cross_entropy_with_logits(logits_fake, train_labels.float())
        loss = loss_real + loss_fake
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        if (epoch+1) % 30 == 0:
            with torch.no_grad():
                attn = F.softmax(logic.attn_logits, dim=0)
                print(f"      Logic epoch {epoch+1}: loss={loss.item():.4f}, attn_obj={attn[0].item():.4f}")

def evaluate(slot_model, logic, feat, labels, device):
    with torch.no_grad():
        Z_list, _ = slot_model(feat.to(device))
        Z_obj, Z_bg = Z_list[0], Z_list[1]
        logits = logic(Z_obj, Z_bg, tau=0.01)
        preds = (logits > 0).long()
        acc = (preds == labels.to(device)).float().mean().item() * 100
    return acc

# ===============================
# ANALYSIS FUNCTIONS (for final seed)
# ===============================
def compute_slot_cosine_similarity(slot_model, feat, device):
    with torch.no_grad():
        Z_list, _ = slot_model(feat.to(device))
        sim = fast_cosine_sim(Z_list[0], Z_list[1]).cpu().numpy()
    return sim.mean(), sim.std()

def plot_attention_distribution(attn_weights, save_path):
    plt.figure(figsize=(6,4))
    plt.hist(attn_weights, bins=30, alpha=0.7, color='royalblue', edgecolor='black')
    plt.xlabel('Attention to object slot')
    plt.ylabel('Frequency')
    plt.title('Slot attention distribution (supervised)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_tsne(z_obj, labels, save_path):
    # z_obj: [N, D] complex, take angles for visualization
    z_np = torch.angle(z_obj).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(z_np)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(embedded[:,0], embedded[:,1], c=labels.cpu().numpy(),
                          cmap='coolwarm', alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Class (even=0, odd=1)')
    plt.title('t-SNE of object slot phasors (supervised)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ===============================
# MAIN LOOP
# ===============================
def main():
    print("="*70)
    print("Experiment 1: Colored MNIST")
    print("="*70)

    # Create datasets
    train_dataset = ColoredMNIST(num_samples=CONFIG['train_samples'],
                                 corr=CONFIG['corr_train'], invert=False)
    iid_dataset = ColoredMNIST(num_samples=CONFIG['test_samples'],
                               corr=CONFIG['corr_test_iid'], invert=False)
    ood_dataset = ColoredMNIST(num_samples=CONFIG['test_samples'],
                               corr=CONFIG['corr_test_ood'], invert=CONFIG['invert_ood'])

    loaders = {
        'train': DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True),
        'iid': DataLoader(iid_dataset, batch_size=CONFIG['batch_size'],
                          shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True),
        'ood': DataLoader(ood_dataset, batch_size=CONFIG['batch_size'],
                          shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    }

    # Feature extractor
    backbone = get_feature_extractor()
    print("Extracting features for training...")
    train_feat, train_labels, train_colors = extract_features(loaders['train'], backbone)
    print("Extracting features for IID test...")
    iid_feat, iid_labels, _ = extract_features(loaders['iid'], backbone)
    print("Extracting features for OOD test...")
    ood_feat, ood_labels, _ = extract_features(loaders['ood'], backbone)

    # Normalize features
    train_feat = F.normalize(train_feat, dim=-1)
    iid_feat = F.normalize(iid_feat, dim=-1)
    ood_feat = F.normalize(ood_feat, dim=-1)

    # Move to device
    train_feat = train_feat.to(device)
    train_labels = train_labels.to(device)
    train_colors = train_colors.to(device)
    iid_feat = iid_feat.to(device)
    iid_labels = iid_labels.to(device)
    ood_feat = ood_feat.to(device)
    ood_labels = ood_labels.to(device)

    # Baseline Logistic Regression (for reference)
    X_train = train_feat.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    baseline_iid = lr.score(scaler.transform(iid_feat.cpu().numpy()), iid_labels.cpu().numpy()) * 100
    baseline_ood = lr.score(scaler.transform(ood_feat.cpu().numpy()), ood_labels.cpu().numpy()) * 100
    print(f"\nBaseline Logistic Regression: IID = {baseline_iid:.1f}%, OOD = {baseline_ood:.1f}%")

    # Storage for results across seeds
    all_results = []

    for seed in range(CONFIG['num_seeds']):
        print(f"\n--- Seed {seed+1}/{CONFIG['num_seeds']} ---")

        # ----- Unsupervised slots -----
        slot_unsup = SlotModel().to(device)
        train_unsupervised_slots(slot_unsup, train_feat, CONFIG['epochs_slot_unsup'], device, seed)
        with torch.no_grad():
            Z_list, _ = slot_unsup(train_feat)
            Z_obj_unsup, Z_bg_unsup = Z_list[0], Z_list[1]
        logic_unsup = CausalLogic().to(device)
        train_causal_logic(logic_unsup, Z_obj_unsup, Z_bg_unsup, train_labels.float(), device, seed)
        acc_iid_unsup = evaluate(slot_unsup, logic_unsup, iid_feat, iid_labels, device)
        acc_ood_unsup = evaluate(slot_unsup, logic_unsup, ood_feat, ood_labels, device)
        attn_obj_unsup = F.softmax(logic_unsup.attn_logits, dim=0)[0].item()
        # Compute slot cosine similarity
        sim_unsup, sim_unsup_std = compute_slot_cosine_similarity(slot_unsup, train_feat, device)

        # ----- Supervised slots -----
        slot_sup = SlotModel().to(device)
        train_supervised_slots(slot_sup, train_feat, train_labels, train_colors, device, seed)
        with torch.no_grad():
            Z_list, _ = slot_sup(train_feat)
            Z_obj_sup, Z_bg_sup = Z_list[0], Z_list[1]
        logic_sup = CausalLogic().to(device)
        train_causal_logic(logic_sup, Z_obj_sup, Z_bg_sup, train_labels.float(), device, seed)
        acc_iid_sup = evaluate(slot_sup, logic_sup, iid_feat, iid_labels, device)
        acc_ood_sup = evaluate(slot_sup, logic_sup, ood_feat, ood_labels, device)
        attn_obj_sup = F.softmax(logic_sup.attn_logits, dim=0)[0].item()
        sim_sup, sim_sup_std = compute_slot_cosine_similarity(slot_sup, train_feat, device)

        # ----- Do‑intervention (supervised) -----
        with torch.no_grad():
            original_attn = logic_sup.attn_logits.clone()
            logic_sup.attn_logits.data = torch.tensor([10.0, -10.0], device=device)
            acc_ood_do = evaluate(slot_sup, logic_sup, ood_feat, ood_labels, device)
            logic_sup.attn_logits.data = original_attn

        # Store results
        seed_result = {
            'seed': seed,
            'unsup_iid': acc_iid_unsup,
            'unsup_ood': acc_ood_unsup,
            'unsup_attn': attn_obj_unsup,
            'unsup_slot_cosine': sim_unsup,
            'sup_iid': acc_iid_sup,
            'sup_ood': acc_ood_sup,
            'sup_attn': attn_obj_sup,
            'sup_slot_cosine': sim_sup,
            'sup_ood_do': acc_ood_do
        }
        all_results.append(seed_result)
        print(f"  Unsup: IID={acc_iid_unsup:.1f}% OOD={acc_ood_unsup:.1f}% attn={attn_obj_unsup:.4f} slot_cos={sim_unsup:.4f}")
        print(f"  Sup:   IID={acc_iid_sup:.1f}% OOD={acc_ood_sup:.1f}% do={acc_ood_do:.1f}% attn={attn_obj_sup:.4f} slot_cos={sim_sup:.4f}")

        # Save the last seed's models for analysis
        if seed == CONFIG['num_seeds'] - 1:
            torch.save(slot_sup.state_dict(), os.path.join(CONFIG['results_dir'], 'models', 'slot_sup_final.pth'))
            torch.save(logic_sup.state_dict(), os.path.join(CONFIG['results_dir'], 'models', 'logic_sup_final.pth'))
            torch.save(slot_unsup.state_dict(), os.path.join(CONFIG['results_dir'], 'models', 'slot_unsup_final.pth'))
            torch.save(logic_unsup.state_dict(), os.path.join(CONFIG['results_dir'], 'models', 'logic_unsup_final.pth'))
            # Store final feature tensors for analysis
            torch.save(train_feat.cpu(), os.path.join(CONFIG['results_dir'], 'train_feat.pt'))
            torch.save(train_labels.cpu(), os.path.join(CONFIG['results_dir'], 'train_labels.pt'))

    # Aggregate statistics
    unsup_iid_mean = np.mean([r['unsup_iid'] for r in all_results])
    unsup_iid_std = np.std([r['unsup_iid'] for r in all_results])
    unsup_ood_mean = np.mean([r['unsup_ood'] for r in all_results])
    unsup_ood_std = np.std([r['unsup_ood'] for r in all_results])
    unsup_attn_mean = np.mean([r['unsup_attn'] for r in all_results])
    unsup_cos_mean = np.mean([r['unsup_slot_cosine'] for r in all_results])

    sup_iid_mean = np.mean([r['sup_iid'] for r in all_results])
    sup_iid_std = np.std([r['sup_iid'] for r in all_results])
    sup_ood_mean = np.mean([r['sup_ood'] for r in all_results])
    sup_ood_std = np.std([r['sup_ood'] for r in all_results])
    sup_ood_do_mean = np.mean([r['sup_ood_do'] for r in all_results])
    sup_ood_do_std = np.std([r['sup_ood_do'] for r in all_results])
    sup_attn_mean = np.mean([r['sup_attn'] for r in all_results])
    sup_cos_mean = np.mean([r['sup_slot_cosine'] for r in all_results])

    print("\n" + "="*70)
    print("Final Results (averaged over {} seeds)".format(CONFIG['num_seeds']))
    print("-"*70)
    print(f"Unsupervised slots: IID = {unsup_iid_mean:.1f}% ± {unsup_iid_std:.1f}%")
    print(f"                     OOD = {unsup_ood_mean:.1f}% ± {unsup_ood_std:.1f}%")
    print(f"                     Attention to object = {unsup_attn_mean:.4f}")
    print(f"                     Slot cosine similarity = {unsup_cos_mean:.4f}")
    print(f"Supervised slots:   IID = {sup_iid_mean:.1f}% ± {sup_iid_std:.1f}%")
    print(f"                     OOD = {sup_ood_mean:.1f}% ± {sup_ood_std:.1f}%")
    print(f"                     After do‑intervention OOD = {sup_ood_do_mean:.1f}% ± {sup_ood_do_std:.1f}%")
    print(f"                     Attention to object = {sup_attn_mean:.4f}")
    print(f"                     Slot cosine similarity = {sup_cos_mean:.4f}")

    # Save results to CSV and JSON
    csv_path = os.path.join(CONFIG['results_dir'], 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    # Convert numpy floats to Python floats for JSON serialization
    summary = {
        'baseline_iid': float(baseline_iid),
        'baseline_ood': float(baseline_ood),
        'unsup_iid_mean': float(unsup_iid_mean), 'unsup_iid_std': float(unsup_iid_std),
        'unsup_ood_mean': float(unsup_ood_mean), 'unsup_ood_std': float(unsup_ood_std),
        'unsup_attn_mean': float(unsup_attn_mean),
        'unsup_slot_cosine_mean': float(unsup_cos_mean),
        'sup_iid_mean': float(sup_iid_mean), 'sup_iid_std': float(sup_iid_std),
        'sup_ood_mean': float(sup_ood_mean), 'sup_ood_std': float(sup_ood_std),
        'sup_ood_do_mean': float(sup_ood_do_mean), 'sup_ood_do_std': float(sup_ood_do_std),
        'sup_attn_mean': float(sup_attn_mean),
        'sup_slot_cosine_mean': float(sup_cos_mean)
    }
    with open(os.path.join(CONFIG['results_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    # ----- Analysis for the last seed (visualizations) -----
    print("\nGenerating analysis plots (last seed)...")
    # Load the last seed's models and data
    slot_sup_last = SlotModel().to(device)
    slot_sup_last.load_state_dict(torch.load(os.path.join(CONFIG['results_dir'], 'models', 'slot_sup_final.pth')))
    logic_sup_last = CausalLogic().to(device)
    logic_sup_last.load_state_dict(torch.load(os.path.join(CONFIG['results_dir'], 'models', 'logic_sup_final.pth')))
    train_feat_last = torch.load(os.path.join(CONFIG['results_dir'], 'train_feat.pt')).to(device)
    train_labels_last = torch.load(os.path.join(CONFIG['results_dir'], 'train_labels.pt')).to(device)

    # Attention distribution on training set
    attn_weights = []
    with torch.no_grad():
        for i in range(0, len(train_feat_last), 256):
            batch = train_feat_last[i:i+256]
            _, probs = slot_sup_last(batch)
            attn_weights.append(probs[:,0,0].cpu().numpy())
    attn_weights = np.concatenate(attn_weights)
    plot_attention_distribution(attn_weights, os.path.join(CONFIG['results_dir'], 'plots', 'attention_distribution.pdf'))

    # t-SNE of object slot phasors (use training data, limit for speed)
    with torch.no_grad():
        Z_list, _ = slot_sup_last(train_feat_last[:2000])
        z_obj = Z_list[0]
    plot_tsne(z_obj, train_labels_last[:2000], os.path.join(CONFIG['results_dir'], 'plots', 'tsne_object_slots.pdf'))

    # Bar plot comparing IID/OOD
    fig, ax = plt.subplots(figsize=(8,5))
    categories = ['IID', 'OOD', 'OOD (do)']
    unsup = [unsup_iid_mean, unsup_ood_mean, 0]
    sup = [sup_iid_mean, sup_ood_mean, sup_ood_do_mean]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, unsup, width, label='Unsupervised slots', color='royalblue', edgecolor='black')
    ax.bar(x + width/2, sup, width, label='Supervised slots', color='darkorange', edgecolor='black')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Colored MNIST: OOD Generalization')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['results_dir'], 'plots', 'accuracy_bars.pdf'), dpi=150)
    plt.savefig(os.path.join(CONFIG['results_dir'], 'plots', 'accuracy_bars.png'), dpi=150)
    plt.close()

    print(f"\nAll results and plots saved to {CONFIG['results_dir']}")

if __name__ == "__main__":
    main()
