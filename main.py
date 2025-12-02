# ============================================================
# TRABALHO M3 - YOLOv11 para detecção de caracteres em placas
# Script único: prepara dataset, treina YOLOv11,
# avalia no conjunto de teste (Precisão, Recall, IoU, F1 e matriz de confusão).
# ============================================================

# 1) Imports principais
import os, random, shutil
from pathlib import Path
import numpy as np
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
from collections import defaultdict

# ------------------------------------------------------------
# CONFIGURAÇÕES DO DATASET
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# PASTA ONDE ESTÃO TODAS AS IMAGENS + ARQUIVOS .TXT ORIGINAIS (M2)
RAW_DIR = BASE_DIR / "dataset"       # <<< NOME EXATO DA PASTA

# PASTA DE SAÍDA NO FORMATO ESPERADO PELO ULTRALYTICS (train/val/test)
YOLO_DIR = BASE_DIR / "placas_yolo"

# Proporções dos splits (por placa)
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15   # o restante vira teste

# extensões aceitas para as imagens
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ------------------------------------------------------------
# 2) COLETA DAS IMAGENS E LABELS, AGRUPANDO POR PLACA
#    - assume que cada imagem tem um .txt com o mesmo nome
#      no formato YOLO: class xc yc w h (normalizado)
#    - define "assinatura" da placa a partir da sequência de
#      classes ordenadas da esquerda para a direita
# ------------------------------------------------------------

assert RAW_DIR.exists(), f"PASTA RAW_DIR não existe: {RAW_DIR}"

def get_plate_signature(label_path: Path):
    """
    Lê o .txt da imagem e retorna uma assinatura imutável da placa
    (tupla com os IDs de classe dos caracteres ordenados da esquerda
    para a direita).
    """
    # encoding explícito pra evitar frescura no Windows
    text = Path(label_path).read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    entries = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc = float(parts[1])   # centro x normalizado (0-1)
        entries.append((xc, cls_id))
    if not entries:
        return ()
    # ordena pelos centros x (esquerda -> direita)
    entries.sort(key=lambda t: t[0])
    signature = tuple(cls for _, cls in entries)
    return signature

print(f"\n[INFO] Procurando imagens em {RAW_DIR} ...")
all_images = [p for p in RAW_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
print(f"[INFO] Encontradas {len(all_images)} imagens com extensão válida.\n")

dataset_items = []   # cada item: {"img": Path, "label": Path, "signature": tuple}
classes_set = set()

for idx, img_path in enumerate(all_images, 1):
    label_path = img_path.with_suffix(".txt")  # mesmo nome, extensão .txt
    if not label_path.exists():
        print(f"[AVISO] sem label para {img_path.name}, ignorando essa imagem.")
        continue

    signature = get_plate_signature(label_path)
    if not signature:
        print(f"[AVISO] label vazio em {label_path.name}, ignorando.")
        continue

    # acumula classes usadas
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    classes_set.add(int(float(parts[0])))
                except ValueError:
                    pass

    dataset_items.append({"img": img_path, "label": label_path, "signature": signature})

    # LOG a cada 500 imagens pra mostrar progresso
    if idx % 500 == 0 or idx == len(all_images):
        print(f"[INFO] Processadas {idx}/{len(all_images)} imagens...")

print(f"\n[INFO] Total de imagens com label: {len(dataset_items)}")

if not dataset_items:
    print("[ERRO] Nenhuma imagem com label foi encontrada. Verifique a pasta 'dataset'.")
    raise SystemExit(1)

# ------------------------------------------------------------
# 3) SPLIT POR PLACA: train / val / test
#    - garante que imagens com a mesma assinatura de placa
#      fiquem sempre no mesmo conjunto
# ------------------------------------------------------------
plates_dict = defaultdict(list)  # assinatura -> [items]
for item in dataset_items:
    plates_dict[item["signature"]].append(item)

unique_signatures = list(plates_dict.keys())
random.seed(42)
random.shuffle(unique_signatures)

n_total_plates = len(unique_signatures)
n_train = int(TRAIN_RATIO * n_total_plates)
n_val   = int(VAL_RATIO   * n_total_plates)
# restante vai para teste
train_sigs = set(unique_signatures[:n_train])
val_sigs   = set(unique_signatures[n_train:n_train + n_val])
test_sigs  = set(unique_signatures[n_train + n_val:])

splits = {"train": [], "val": [], "test": []}
for sig, items in plates_dict.items():
    if sig in train_sigs:
        split = "train"
    elif sig in val_sigs:
        split = "val"
    else:
        split = "test"
    splits[split].extend(items)

print("\n[INFO] Tamanho dos splits (por imagem):")
for split_name, items in splits.items():
    print(f"  {split_name}: {len(items)} imagens")

# ------------------------------------------------------------
# 4) ORGANIZA ARQUIVOS NO FORMATO YOLO (imagens/labels por split)
# ------------------------------------------------------------
print("\n[INFO] Organizando dataset no formato YOLO em:", YOLO_DIR)

# limpa/Cria pasta de saída (com tolerância ao Windows frescurento)
if YOLO_DIR.exists():
    try:
        print(f"[INFO] Apagando pasta antiga: {YOLO_DIR}")
        shutil.rmtree(YOLO_DIR)
    except Exception as e:
        print(f"[AVISO] Não consegui apagar completamente {YOLO_DIR}: {e}")
        print("[AVISO] Vou reaproveitar a pasta existente e sobrescrever arquivos quando copiar.")

YOLO_DIR.mkdir(parents=True, exist_ok=True)


for split_name, items in splits.items():
    img_out_dir = YOLO_DIR / "images" / split_name
    lbl_out_dir = YOLO_DIR / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        dst_img = img_out_dir / item["img"].name
        dst_lbl = lbl_out_dir / item["label"].name
        shutil.copy2(item["img"], dst_img)
        shutil.copy2(item["label"], dst_lbl)

print(f"[INFO] Dataset organizado em: {YOLO_DIR}")

# ------------------------------------------------------------
# 5) CRIA ARQUIVO YAML PARA O ULTRALYTICS
#    - mapeando classes 0-9, A-Z (se existirem)
# ------------------------------------------------------------
max_class_id = max(classes_set) if classes_set else 0
num_classes = max_class_id + 1

CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
names = {}
for i in range(num_classes):
    if i < len(CHARS):
        names[i] = CHARS[i]
    else:
        names[i] = f"cls_{i}"

data_yaml = {
    "path": str(YOLO_DIR),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": names,
}

yaml_path = YOLO_DIR / "placas.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)

print("\n[INFO] Arquivo YAML criado em:", yaml_path)
print("----------------------------------------------")
print(yaml_path.read_text())
print("----------------------------------------------\n")

# ------------------------------------------------------------
# 6) TREINA MODELO YOLOv11
# ------------------------------------------------------------
print("[INFO] Carregando modelo YOLO11n...")
model = YOLO("yolo11n.pt")

print("[INFO] Iniciando treino...")
results_train = model.train(
    data=str(yaml_path),
    epochs=25,          # mais leve que 80 pra CPU
    imgsz=640,
    batch=8,            # batch menor ajuda bastante em CPU
    patience=10,        # early stopping
    project="runs_placas",
    name="yolo11_chars",
    workers=0,          # importante no Windows para evitar problemas de multiprocessing
    device="cpu",       # força usar CPU
)

# Caminho do melhor modelo salvo
best_model_path = Path("runs_placas") / "yolo11_chars" / "weights" / "best.pt"
print("\n[INFO] Melhor modelo salvo em:", best_model_path)

# Recarrega melhor modelo treinado
best_model = YOLO(str(best_model_path))

# ------------------------------------------------------------
# 7) AVALIAÇÃO NO CONJUNTO DE TESTE
#    - Usa métricas internas do Ultralytics (mAP, precision, recall)
#    - e métricas manuais por caractere (IoU médio, F1, matriz de confusão)
# ------------------------------------------------------------

print("\n[INFO] Rodando avaliação no conjunto de TEST...")

# 7.1) Validação padrão Ultralytics com gráficos
metrics_test = best_model.val(
    data=str(yaml_path),
    split="test",
    imgsz=640,
    conf=0.25,
    iou=0.5,
    plots=True,   # gera confusion_matrix.png, PR-curves etc.
    workers=0,
    device="cpu",
)

print("\n--- Métricas Ultralytics (Detecção) ---")
print("mAP50-95:", float(metrics_test.box.map))
print("mAP50:   ", float(metrics_test.box.map50))
print("mAP75:   ", float(metrics_test.box.map75))
print("Precisão média (mp):", float(metrics_test.box.mp))
print("Recall médio (mr):   ", float(metrics_test.box.mr))
f1_det = 2 * float(metrics_test.box.mp) * float(metrics_test.box.mr) / (float(metrics_test.box.mp) + float(metrics_test.box.mr) + 1e-8)
print("F1-score (detecção, média das classes):", f1_det)

# 7.2) Avaliação manual por caractere: IoU médio + matriz de confusão de classes
def yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h):
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
    y_true = []
    y_pred = []
    ious = []

    image_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

    for img_path in image_files:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground truth
        gt_boxes = []
        with open(lbl_path, "r", encoding="utf-8") as f:
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

        # Predições do modelo
        results = model(img)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        pred_xyxy = boxes.xyxy.cpu().numpy()
        pred_cls  = boxes.cls.cpu().numpy().astype(int)
        pred_conf = boxes.conf.cpu().numpy()

        # filtra por confiança
        mask = pred_conf >= conf_thr
        pred_xyxy = pred_xyxy[mask]
        pred_cls  = pred_cls[mask]

        if len(pred_cls) == 0:
            continue

        used_pred = set()
        # Matching simples: para cada GT, pega a predição com maior IoU
        for cls_gt, box_gt in gt_boxes:
            best_iou = 0.0
            best_j = None
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

test_img_dir = YOLO_DIR / "images" / "test"
test_lbl_dir = YOLO_DIR / "labels" / "test"

mean_iou, y_true, y_pred = evaluate_on_split(
    best_model, test_img_dir, test_lbl_dir,
    conf_thr=0.25, iou_match_thr=0.5,
)

print("\n--- Avaliação manual por caractere (TEST) ---")
print(f"IoU médio dos matches (IoU >= 0.5): {mean_iou:.4f}")

if len(y_true) == 0:
    print("Nenhum match GT-pred encontrado. Verifique limiares de conf/iou ou dataset.")
else:
    # classes realmente usadas no conjunto de teste
    used_labels = sorted(set(y_true.tolist()))
    target_names = [names[i] if i in names else f"cls_{i}" for i in used_labels]

    print("\nRelatório de classificação por caractere:")
    print(classification_report(
        y_true, y_pred,
        labels=used_labels,
        target_names=target_names,
        zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=used_labels)
    print("Matriz de confusão (linhas = verdadeiro, colunas = predito):")
    print("Índices das classes (na ordem das linhas/colunas):", used_labels)
    print(cm)

    # métricas macro (média entre classes)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nPrecisão macro: {prec_macro:.4f}")
    print(f"Recall   macro: {rec_macro:.4f}")
    print(f"F1-score macro: {f1_macro:.4f}")

print("\nFIM DO SCRIPT.")
