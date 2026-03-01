"""
異常検知パイプライン
====================
フロー：
1. 画像を無作為にインプット
2. Isolation Forest / LOF により異常判定
3. 異常領域をクロップ（グラディエントベース）
4. CLIP で特徴量抽出
5. UMAP で次元削減・可視化
6. HDBSCAN でクラスタ分類
7. 各クラスタ = 異常パターン候補として出力

必要パッケージのインストール：
    pip install torch torchvision transformers scikit-learn umap-learn hdbscan pillow matplotlib numpy

使い方：
    python anomaly_detection.py --input_dir ./images --output_dir ./results
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI不要のバックエンド
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFilter
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import umap
import hdbscan


# ============================================================
# 設定
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ============================================================
# Step 1: 画像の読み込み
# ============================================================
def load_images(input_dir: str) -> tuple[list[Image.Image], list[str]]:
    """指定ディレクトリから画像を読み込む"""
    input_path = Path(input_dir)
    image_paths = [
        p for p in input_path.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise ValueError(f"画像が見つかりません: {input_dir}")

    images, paths = [], []
    for p in sorted(image_paths):
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            paths.append(str(p))
            print(f"  読み込み: {p.name}")
        except Exception as e:
            print(f"  スキップ ({p.name}): {e}")

    print(f"\n合計 {len(images)} 枚の画像を読み込みました\n")
    return images, paths


# ============================================================
# Step 2: 基本特徴量の抽出（異常判定用）
# ============================================================
def extract_basic_features(images: list[Image.Image]) -> np.ndarray:
    """
    Isolation Forest / LOF 用の簡易特徴量を抽出する
    ・リサイズしてフラット化（ピクセル統計）
    ・明るさ・コントラスト・エッジ量を追加
    """
    features = []
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    for img in images:
        tensor = transform(img).numpy()  # (3, 64, 64)

        # ピクセル統計
        mean = tensor.mean(axis=(1, 2))     # チャネルごとの平均
        std  = tensor.std(axis=(1, 2))      # チャネルごとの標準偏差

        # エッジ量（Sobelフィルタの代わりにグレースケールでLaplacian）
        gray = img.convert("L").resize((64, 64))
        gray_arr = np.array(gray, dtype=np.float32) / 255.0
        edge = np.abs(np.gradient(gray_arr)).mean()

        # 明るさ・コントラスト
        brightness = gray_arr.mean()
        contrast   = gray_arr.std()

        feat = np.concatenate([mean, std, [edge, brightness, contrast]])
        features.append(feat)

    return np.array(features)


# ============================================================
# Step 2: 異常判定（Isolation Forest + LOF）
# ============================================================
def detect_anomalies(
    features: np.ndarray,
    method: str = "both",
    contamination: float = 0.3
) -> np.ndarray:
    """
    異常判定を行い、異常フラグ配列を返す（True = 異常）

    Parameters
    ----------
    features      : 特徴量行列
    method        : 'isolation_forest' / 'lof' / 'both'（多数決）
    contamination : 異常とみなす割合（0〜0.5）
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    if method in ("isolation_forest", "both"):
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso_labels = iso.fit_predict(X)  # -1=異常, 1=正常
        iso_anomaly = iso_labels == -1

    if method in ("lof", "both"):
        n_neighbors = min(20, len(X) - 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        lof_labels = lof.fit_predict(X)  # -1=異常, 1=正常
        lof_anomaly = lof_labels == -1

    if method == "isolation_forest":
        anomaly_flags = iso_anomaly
    elif method == "lof":
        anomaly_flags = lof_anomaly
    else:
        # 多数決：両方が異常と判定した場合のみ異常
        anomaly_flags = iso_anomaly & lof_anomaly

    n_anomaly = anomaly_flags.sum()
    print(f"異常判定結果: {n_anomaly} / {len(features)} 枚を異常と判定\n")
    return anomaly_flags


# ============================================================
# Step 3: 異常領域のクロップ
# ============================================================
def crop_anomaly_regions(
    images: list[Image.Image],
    anomaly_flags: np.ndarray,
    crop_size: int = 128
) -> tuple[list[Image.Image], list[int]]:
    """
    異常画像から「最も異常らしい領域」をクロップする
    ・グラジエント（エッジ強度）が最大の領域を異常領域と仮定
    ・sliding windowで最大エッジ領域を探索
    """
    cropped_images = []
    original_indices = []

    for idx, (img, is_anomaly) in enumerate(zip(images, anomaly_flags)):
        if not is_anomaly:
            continue

        w, h = img.size
        if w < crop_size or h < crop_size:
            # 画像が小さい場合はそのままリサイズ
            cropped = img.resize((crop_size, crop_size))
            cropped_images.append(cropped)
            original_indices.append(idx)
            continue

        # グレースケールでエッジマップを生成
        gray = img.convert("L")
        edge_map = gray.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edge_map, dtype=np.float32)

        # sliding windowで最大エッジ領域を探索
        best_score = -1
        best_box = (0, 0, crop_size, crop_size)

        step = crop_size // 4  # ストライド
        for y in range(0, h - crop_size + 1, step):
            for x in range(0, w - crop_size + 1, step):
                score = edge_arr[y:y+crop_size, x:x+crop_size].mean()
                if score > best_score:
                    best_score = score
                    best_box = (x, y, x + crop_size, y + crop_size)

        cropped = img.crop(best_box)
        cropped_images.append(cropped)
        original_indices.append(idx)

    print(f"異常領域クロップ完了: {len(cropped_images)} 枚\n")
    return cropped_images, original_indices


# ============================================================
# Step 4: CLIP で特徴量抽出
# ============================================================
def extract_clip_features(
    images: list[Image.Image],
    model_name: str = CLIP_MODEL_NAME,
    batch_size: int = 16
) -> np.ndarray:
    """CLIPのビジョンエンコーダで特徴量を抽出する"""
    print(f"CLIPモデルを読み込み中: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = F.normalize(feats, dim=-1)  # L2正規化

        all_features.append(feats.cpu().numpy())
        print(f"  CLIP特徴量抽出: {min(i+batch_size, len(images))}/{len(images)} 枚")

    features = np.vstack(all_features)
    print(f"\nCLIP特徴量shape: {features.shape}\n")
    return features


# ============================================================
# Step 5: UMAP で次元削減
# ============================================================
def reduce_with_umap(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1
) -> np.ndarray:
    """UMAPで高次元特徴量を2次元に削減する"""
    n_neighbors = min(n_neighbors, len(features) - 1)
    print(f"UMAP次元削減中: {features.shape[1]}次元 → {n_components}次元")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        metric="cosine"
    )
    embedding = reducer.fit_transform(features)
    print(f"UMAP完了: {embedding.shape}\n")
    return embedding


# ============================================================
# Step 6: HDBSCAN でクラスタ分類
# ============================================================
def cluster_with_hdbscan(
    embedding: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 1
) -> np.ndarray:
    """HDBSCANでクラスタリングを行う（-1はノイズ）"""
    min_cluster_size = max(2, min(min_cluster_size, len(embedding) // 2))
    print(f"HDBSCANクラスタリング中 (min_cluster_size={min_cluster_size})")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(embedding)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"クラスタ数: {n_clusters} （ノイズ: {n_noise} 枚）\n")
    return labels


# ============================================================
# Step 7: 結果の可視化・保存
# ============================================================
def visualize_results(
    embedding: np.ndarray,
    labels: np.ndarray,
    cropped_images: list[Image.Image],
    original_indices: list[int],
    image_paths: list[str],
    output_dir: str
):
    """UMAP散布図・クラスタ別サムネイル・サマリーを保存する"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])

    # カラーマップ
    cmap = plt.cm.get_cmap("tab10", max(n_clusters, 1))
    colors = {
        label: ("gray" if label == -1 else cmap(label))
        for label in unique_labels
    }

    # ── 散布図 ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    for label in unique_labels:
        mask = labels == label
        color = colors[label]
        name = "ノイズ" if label == -1 else f"クラスタ {label}"
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[color], label=name, s=120, alpha=0.85,
            edgecolors="white", linewidths=0.5
        )
        # クラスタ中心にラベルを表示
        if label >= 0:
            cx = embedding[mask, 0].mean()
            cy = embedding[mask, 1].mean()
            ax.text(cx, cy, str(label), color="white",
                    fontsize=12, fontweight="bold", ha="center", va="center")

    ax.set_title("異常パターン UMAP可視化", color="white", fontsize=16, pad=15)
    ax.set_xlabel("UMAP-1", color="#aaaaaa")
    ax.set_ylabel("UMAP-2", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333344")
    legend = ax.legend(facecolor="#1a1a2e", edgecolor="#333344", labelcolor="white")
    plt.tight_layout()
    scatter_path = output_path / "umap_scatter.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"散布図保存: {scatter_path}")

    # ── クラスタ別サムネイル ──────────────────────────
    for label in unique_labels:
        if label == -1:
            cluster_name = "noise"
        else:
            cluster_name = f"cluster_{label:02d}"

        mask_indices = [i for i, l in enumerate(labels) if l == label]
        cluster_imgs = [cropped_images[i] for i in mask_indices]

        if not cluster_imgs:
            continue

        n = len(cluster_imgs)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        thumb_size = 128

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2),
                                  facecolor="#0f0f1a")
        fig.suptitle(
            f"{'ノイズ' if label == -1 else f'クラスタ {label}'} ({n}枚)",
            color="white", fontsize=14
        )

        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
        for ax_i, (ax, img) in enumerate(zip(axes_flat, cluster_imgs)):
            ax.imshow(img.resize((thumb_size, thumb_size)))
            orig_idx = original_indices[mask_indices[ax_i]]
            ax.set_title(Path(image_paths[orig_idx]).name,
                         color="#aaaaaa", fontsize=7)
            ax.axis("off")

        for ax in axes_flat[len(cluster_imgs):]:
            ax.axis("off")

        plt.tight_layout()
        thumb_path = output_path / f"{cluster_name}_thumbnails.png"
        plt.savefig(thumb_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"サムネイル保存: {thumb_path}")

    # ── テキストサマリー ──────────────────────────────
    summary_lines = [
        "=" * 50,
        "異常検知パイプライン サマリー",
        "=" * 50,
        f"異常画像数   : {len(cropped_images)} 枚",
        f"クラスタ数   : {n_clusters}",
        f"ノイズ数     : {(labels == -1).sum()} 枚",
        "",
        "クラスタ別ファイル一覧:",
    ]
    for label in unique_labels:
        mask_indices = [i for i, l in enumerate(labels) if l == label]
        name = "ノイズ" if label == -1 else f"クラスタ {label}"
        summary_lines.append(f"\n[{name}] {len(mask_indices)}枚")
        for i in mask_indices:
            orig_idx = original_indices[i]
            summary_lines.append(f"  - {Path(image_paths[orig_idx]).name}")

    summary_text = "\n".join(summary_lines)
    summary_path = output_path / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\nサマリー保存: {summary_path}")
    print("\n" + summary_text)


# ============================================================
# メインパイプライン
# ============================================================
def run_pipeline(
    input_dir: str,
    output_dir: str,
    method: str = "both",
    contamination: float = 0.3,
    crop_size: int = 128,
    umap_neighbors: int = 15,
    min_cluster_size: int = 2,
):
    print("=" * 50)
    print("異常検知パイプライン 開始")
    print("=" * 50 + "\n")

    # Step 1: 画像読み込み
    print("--- Step 1: 画像読み込み ---")
    images, image_paths = load_images(input_dir)

    if len(images) < 3:
        raise ValueError("画像が3枚以上必要です。")

    # Step 2: 異常判定
    print("--- Step 2: 異常判定 ---")
    basic_features = extract_basic_features(images)
    anomaly_flags = detect_anomalies(basic_features, method=method,
                                      contamination=contamination)

    if anomaly_flags.sum() == 0:
        print("異常画像が検出されませんでした。contaminationの値を上げてください。")
        return

    # Step 3: 異常領域クロップ
    print("--- Step 3: 異常領域クロップ ---")
    cropped_images, original_indices = crop_anomaly_regions(
        images, anomaly_flags, crop_size=crop_size
    )

    if len(cropped_images) == 0:
        print("クロップできる画像がありませんでした。")
        return

    # Step 4: CLIP特徴量抽出
    print("--- Step 4: CLIP特徴量抽出 ---")
    clip_features = extract_clip_features(cropped_images)

    # Step 5: UMAP次元削減
    print("--- Step 5: UMAP次元削減 ---")
    embedding = reduce_with_umap(clip_features, n_neighbors=umap_neighbors)

    # Step 6: HDBSCANクラスタリング
    print("--- Step 6: HDBSCANクラスタリング ---")
    cluster_labels = cluster_with_hdbscan(
        embedding, min_cluster_size=min_cluster_size
    )

    # Step 7: 結果可視化・保存
    print("--- Step 7: 結果保存 ---")
    visualize_results(
        embedding, cluster_labels, cropped_images,
        original_indices, image_paths, output_dir
    )

    print("\n" + "=" * 50)
    print("パイプライン完了")
    print("=" * 50)


# ============================================================
# エントリポイント
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="異常検知パイプライン")
    parser.add_argument("--input_dir",       type=str,   default="./images",
                        help="入力画像ディレクトリ")
    parser.add_argument("--output_dir",      type=str,   default="./results",
                        help="結果出力ディレクトリ")
    parser.add_argument("--method",          type=str,   default="both",
                        choices=["isolation_forest", "lof", "both"],
                        help="異常判定手法")
    parser.add_argument("--contamination",   type=float, default=0.3,
                        help="異常とみなす割合 (0〜0.5)")
    parser.add_argument("--crop_size",       type=int,   default=128,
                        help="クロップサイズ（ピクセル）")
    parser.add_argument("--umap_neighbors",  type=int,   default=15,
                        help="UMAPのn_neighbors")
    parser.add_argument("--min_cluster_size",type=int,   default=2,
                        help="HDBSCANの最小クラスタサイズ")
    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        method=args.method,
        contamination=args.contamination,
        crop_size=args.crop_size,
        umap_neighbors=args.umap_neighbors,
        min_cluster_size=args.min_cluster_size,
    )
