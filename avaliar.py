# avaliar.py
# ------------------------------------------------------------
# Avaliação do modelo YOLOv11 já treinado:
# - usa best.pt
# - calcula métricas Ultralytics (mAP, precisão, recall, F1)
# - calcula IoU médio, relatório de classificação e matriz de confusão
#   no conjunto de TESTE
# ------------------------------------------------------------

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml
from ultralytics import YOLO
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# Caminhos base
BASE_DIR = Path(__file__).resolve().parent
YOLO_DIR = BASE_DIR / "placas_yolo"
yaml_path = YOLO_DIR / "placas.yaml"
best_model_path = BASE_DIR / "runs_placas" / "yolo11_chars" / "weights" / "best.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

assert yaml_path.exists(), f"YAML não encontrado: {yaml_path}"
assert best_model_path.exists(), f"Pesos best.pt não encontrados: {best_model_path}"

# Carrega nomes das classes
with open(yaml_path, "r") as f:
    data_yaml = yaml.safe_load(f)

raw_names = data_yaml.get("names", {})
names = {int(k): str(v) for k, v in raw_names.items()}

print("[INFO] Carregando modelo treinado...")
model = YOLO(str(best_model_path))

# ------------------------------------------------------------
# 1) Métricas padrão Ultralytics no split TEST
# ------------------------------------------------------------
print("[INFO] Avaliando com val() da Ultralytics (split=test)...")
metrics_test = model.val(
    data=str(yaml_path),
    split="test",
    imgsz=640,
    conf=0.25,
    iou=0.5,
    plots=True,   # gera os gráficos, inclusive confusion_matrix.png
    project=str(BASE_DIR / "runs_placas"),
    name="eval_test",
    exist_ok=True,
)

print("\n--- Métricas Ultralytics (Detecção) ---")
print(f"mAP50-95: {float(metrics_test.box.map):.4f}")
print(f"mAP50:    {float(metrics_test.box.map50):.4f}")
print(f"mAP75:    {float(metrics_test.box.map75):.4f}")
print(f"Precisão média (mp): {float(metrics_test.box.mp):.4f}")
print(f"Recall médio (mr):   {float(metrics_test.box.mr):.4f}")
f1_det = 2 * float(metrics_test.box.mp) * float(metrics_test.box.mr) / (
    float(metrics_test.box.mp) + float(metrics_test.box.mr) + 1e-8
)
print(f"F1-score (detecção, média das classes): {f1_det:.4f}")

# ------------------------------------------------------------
# 2) Avaliação manual por caractere (IoU + matriz de confusão)
# ------------------------------------------------------------
def yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h):
    # CORREÇÃO: usar bh no eixo Y (antes estava bw)
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h
    return x1, y1, x2, y2

def box_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-9
    return inter / union

def evaluate_on_split(model, images_dir: Path, labels_dir: Path,
                      conf_thr: float = 0.25, iou_match_thr: float = 0.5):
    y_true, y_pred, ious = [], [], []

    image_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

    for img_path in image_files:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # GT
        gt_boxes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:])
                x1g, y1g, x2g, y2g = yolo_to_xyxy(xc, yc, bw, bh, w, h)
                gt_boxes.append((cls_id, (x1g, y1g, x2g, y2g)))

        if not gt_boxes:
            continue

        # Predições
        results = model(img)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        pred_xyxy = boxes.xyxy.cpu().numpy()
        pred_cls  = boxes.cls.cpu().numpy().astype(int)
        pred_conf = boxes.conf.cpu().numpy()

        mask = pred_conf >= conf_thr
        pred_xyxy = pred_xyxy[mask]
        pred_cls  = pred_cls[mask]

        if len(pred_cls) == 0:
            continue

        used_pred = set()
        for cls_gt, box_gt in gt_boxes:
            best_iou, best_j = 0.0, None
            for j, (box_p, cls_p) in enumerate(zip(pred_xyxy, pred_cls)):
                if j in used_pred:
                    continue
                iou = box_iou_xyxy(box_gt, box_p)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j is not None and best_iou >= iou_match_thr:
                used_pred.add(best_j)
                y_true.append(cls_gt)
                y_pred.append(int(pred_cls[best_j]))
                ious.append(best_iou)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    return mean_iou, y_true, y_pred

print("\n[INFO] Avaliação manual por caractere no TEST...")
test_img_dir = YOLO_DIR / "images" / "test"
test_lbl_dir = YOLO_DIR / "labels" / "test"

mean_iou, y_true, y_pred = evaluate_on_split(
    model, test_img_dir, test_lbl_dir,
    conf_thr=0.25, iou_match_thr=0.5,
)

print(f"IoU médio dos matches (IoU >= 0.5): {mean_iou:.4f}")

if len(y_true) == 0:
    print("Nenhum match GT-pred encontrado. Verifique limiares de conf/iou ou dataset.")
else:
    used_labels = sorted(set(y_true.tolist()))
    target_names = [names.get(i, f"cls_{i}") for i in used_labels]

    print("\nRelatório de classificação por caractere:")
    print(classification_report(
        y_true, y_pred,
        labels=used_labels,
        target_names=target_names,
        zero_division=0,
    ))

    # >>> AQUI FALTAVA: criar a matriz de confusão <<<
    cm = confusion_matrix(y_true, y_pred, labels=used_labels)

    # cria pasta para salvar, reaproveitando a mesma da eval_test
    cm_dir = BASE_DIR / "runs_placas" / "eval_test"
    cm_dir.mkdir(parents=True, exist_ok=True)
    cm_path = cm_dir / "confusion_matrix_manual.png"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(used_labels)))
    ax.set_yticks(np.arange(len(used_labels)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)

    ax.set_ylabel("Verdadeiro")
    ax.set_xlabel("Predito")
    ax.set_title("Matriz de Confusão (avaliação manual)")

    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close(fig)

    print("Matriz de confusão manual salva em:", cm_path)

    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nPrecisão macro: {prec_macro:.4f}")
    print(f"Recall   macro: {rec_macro:.4f}")
    print(f"F1-score macro: {f1_macro:.4f}")

print("\nFIM DA AVALIAÇÃO.")
