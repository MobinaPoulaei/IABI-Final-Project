from model.network import Classifier_1fc, DimReduction
from model.Attention import Attention_Gated as Attention
from model.Attention import Attention_with_Classifier
from model.feature_extraction import resnet50_baseline
import argparse
import torch
import torchvision.models as models
import pandas as pd

from utils.CQ500ImageDistillationDataset import (
    CQ500ImageDistillationDataset,
)  # <-- CHANGED
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)  # Ensure these are imported
import numpy as np
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="CQ500 Distillation")

parser.add_argument("--name", default="cq500_distillation", type=str)
parser.add_argument("--EPOCH", default=2, type=int)  # <-- CHANGED: More epochs for OCT
parser.add_argument("--epoch_step", default="[100]", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--isPar", default=False, type=bool)
parser.add_argument("--log_dir", default="./debug_log", type=str)
parser.add_argument("--train_show_freq", default=40, type=int)
parser.add_argument("--droprate", default="0", type=float)
parser.add_argument("--droprate_2", default="0", type=float)
parser.add_argument("--lr", default=1e-4, type=float)  # <-- CHANGED: Higher LR for OCT
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--lr_decay_ratio", default=0.2, type=float)
parser.add_argument("--batch_size", default=16, type=int)  # <-- CHANGED: Smaller batch
parser.add_argument("--batch_size_v", default=1, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--num_cls", default=4, type=int)  # <-- CHANGED: 4 classes for OCT
parser.add_argument(
    "--data_root", default="/kaggle/working/CQ500_ICH_VS_NORMAL", type=str
)  # <-- CHANGED
parser.add_argument(
    "--checkpoint_path", default="best_model_cq500.pth", type=str
)  # <-- CHANGED
parser.add_argument("--numGroup", default=5, type=int)
parser.add_argument("--total_instance", default=4, type=int)
parser.add_argument("--numGroup_test", default=4, type=int)
parser.add_argument("--total_instance_test", default=4, type=int)
parser.add_argument("--mDim", default=512, type=int)
parser.add_argument("--grad_clipping", default=5, type=float)
parser.add_argument("--isSaveModel", action="store_false")
parser.add_argument("--debug_DATA_dir", default="", type=str)
parser.add_argument("--numLayer_Res", default=0, type=int)
parser.add_argument("--temperature", default=1, type=float)
parser.add_argument("--num_MeanInference", default=1, type=int)
parser.add_argument("--distill_type", default="AFS", type=str)
parser.add_argument("--meta_df_path", default="/kaggle/working/meta_df.csv", type=str)
params = parser.parse_args()

BETA = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==================== TEACHER MODELS (Frozen) ====================
print("Loading Teacher Models...")
model_teacher = resnet50_baseline(True).to(device)
for param in model_teacher.parameters():
    param.requires_grad = False
model_teacher.eval()

classifier_teacher = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(
    params.device
)
attention1_teacher = Attention(params.mDim).to(params.device)
classifier1_teacher = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(
    params.device
)
dimReduction_teacher = DimReduction(
    1024, params.mDim, numLayer_Res=params.numLayer_Res
).to(params.device)
attCls = Attention_with_Classifier(
    L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2
).to(params.device)

# Load checkpoint from Step 1
checkpoint = torch.load(params.checkpoint_path, map_location="cpu", weights_only=False)
classifier_teacher.load_state_dict(
    {
        "fc.weight": checkpoint["att_classifier"]["classifier.fc.weight"],
        "fc.bias": checkpoint["att_classifier"]["classifier.fc.bias"],
    }
)
attCls.load_state_dict(checkpoint["att_classifier"])
attention_teacher = attCls.attention
dimReduction_teacher.load_state_dict(checkpoint["dim_reduction"])
attention1_teacher.load_state_dict(checkpoint["attention"])
classifier1_teacher.load_state_dict(checkpoint["classifier"])

# Freeze teacher
for param in classifier_teacher.parameters():
    param.requires_grad = False
for param in dimReduction_teacher.parameters():
    param.requires_grad = False
for param in attention1_teacher.parameters():
    param.requires_grad = False
for param in classifier1_teacher.parameters():
    param.requires_grad = False
for param in attention_teacher.parameters():
    param.requires_grad = False

classifier_teacher.eval()
dimReduction_teacher.eval()
attention1_teacher.eval()
classifier1_teacher.eval()
attention_teacher.eval()

print("Teacher models loaded and frozen.")

# ==================== STUDENT MODELS (Trainable) ====================
print("Initializing Student Models...")
model_student = resnet50_baseline(True).to(device)
classifier_student = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(
    params.device
)
dimReduction_student = DimReduction(
    1024, params.mDim, numLayer_Res=params.numLayer_Res
).to(params.device)

# Initialize student from teacher weights
classifier_student.load_state_dict(
    {
        "fc.weight": checkpoint["att_classifier"]["classifier.fc.weight"],
        "fc.bias": checkpoint["att_classifier"]["classifier.fc.bias"],
    }
)
dimReduction_student.load_state_dict(checkpoint["dim_reduction"])

print("Student models initialized.")

# ==================== DATASET & DATALOADER ====================
meta_df = pd.read_csv(params.meta_df_path)
trainset = CQ500ImageDistillationDataset(
    path=params.data_root, meta_df=meta_df, mode="train"
)
valset = CQ500ImageDistillationDataset(
    path=params.data_root, meta_df=meta_df, mode="val"
)
testset = CQ500ImageDistillationDataset(
    path=params.data_root, meta_df=meta_df, mode="test"
)

print(f"Train samples: {len(trainset)}")
print(f"Val samples: {len(valset)}")
print(f"Test samples: {len(testset)}")


trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=params.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=params.num_workers,
)
valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=params.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=params.num_workers,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=params.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=params.num_workers,
)

# ==================== OPTIMIZER & SCHEDULER ====================
optimizer = torch.optim.Adam(
    list(model_student.parameters())
    + list(dimReduction_student.parameters())
    + list(classifier_student.parameters()),
    lr=params.lr,
    weight_decay=params.weight_decay,
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, [10, 15], gamma=params.lr_decay_ratio
)

# ==================== LOSS FUNCTION ====================
ce_cri = torch.nn.KLDivLoss(reduction="none").to(params.device)


def min_max_norm(tAA):
    """Normalize attention scores to [0, 1]"""
    return (tAA - torch.min(tAA)) / (torch.max(tAA) - torch.min(tAA) + 1e-8)


# ==================== VALIDATION FUNCTION ====================
def TestModel(test_loader):
    model_student.eval()
    dimReduction_student.eval()
    classifier_student.eval()

    y_score = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, inputs1, inputs2 = data
            inputs = inputs.type(torch.FloatTensor).to(params.device)
            inputs1 = inputs1.type(torch.FloatTensor).to(params.device)

            # Teacher predictions (soft labels)
            inputs_tensor_teacher = model_teacher(inputs)
            tmidFeat_teacher = dimReduction_teacher(inputs_tensor_teacher)
            tPredict_teacher = classifier_teacher(tmidFeat_teacher)
            labels_soft = torch.softmax(tPredict_teacher, -1)

            # Student predictions
            inputs_tensor = model_student(inputs1)
            tmidFeat = dimReduction_student(inputs_tensor)
            tPredict = classifier_student(tmidFeat)
            gSlidePred = torch.softmax(tPredict, dim=1)

            pred = gSlidePred.cpu().data.numpy()
            true = labels_soft.cpu().data.numpy()

            y_score.extend(pred.tolist())
            y_true.extend(true.tolist())
            # y_pred.extend(np.argmax(pred, axis=1).tolist())

    # Calculate metrics
    y_score = np.array(y_score)
    y_true_2d = np.array(y_true)
    y_true = np.argmax(y_true_2d, axis=1)

    y_pred = np.argmax(y_score, axis=1)
    y_prob = y_score[:, 1]
    acc = np.mean(y_true == y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro")

    mae = np.mean(np.abs(y_score - y_true_2d))
    print(
        f"Val result: MAE:{mae:.4f}, ACC:{acc:.4f}, Prec:{precision:.4f}, Recall:{recall:.4f} F1:{f1:.4f}, AUC:{auc:.4f}"
    )
    return mae, acc, f1, auc


# ==================== TRAINING LOOP ====================
best_mae = 1.0
best_acc = 0.0

print("\n" + "=" * 50)
print("Starting Distillation Training...")
print("=" * 50 + "\n")

for epoch in range(params.EPOCH):
    model_student.train()
    dimReduction_student.train()
    classifier_student.train()

    for param_group in optimizer.param_groups:
        curLR = param_group["lr"]
        print(f"Epoch [{epoch+1}/{params.EPOCH}] - Learning Rate: {curLR}")

    total_loss = 0
    total_loss_c = 0
    total_loss_w1 = 0
    total_loss_w2 = 0

    for i, data in enumerate(trainloader):
        inputs, inputs1, inputs2 = data
        inputs = inputs.type(torch.FloatTensor).to(params.device)
        inputs1 = inputs1.type(torch.FloatTensor).to(params.device)
        inputs2 = inputs2.type(torch.FloatTensor).to(params.device)

        # ========== TEACHER FORWARD (No Gradient) ==========
        with torch.no_grad():
            inputs_tensor_teacher = model_teacher(inputs)
            tmidFeat_teacher = dimReduction_teacher(inputs_tensor_teacher)

            # Compute attention scores
            attention_score1 = attention1_teacher(tmidFeat_teacher).squeeze(0)
            attention_score1 = min_max_norm(attention_score1)
            attention_score = attention_teacher(tmidFeat_teacher).squeeze(0)
            attention_score = min_max_norm(attention_score)

            # Combine attention scores
            attention_score = attention_score * attention_score1
            attention_score = abs((2 * attention_score - 1) ** BETA)

            # Teacher predictions
            tPredict_teacher = 0.7 * classifier_teacher(
                tmidFeat_teacher
            ) + 0.3 * classifier1_teacher(tmidFeat_teacher)

        # ========== STUDENT FORWARD (With Gradient) ==========
        inputs_tensor_student = model_student(inputs1)
        tmidFeat_student = dimReduction_student(inputs_tensor_student)
        tPredict_student = classifier_student(tmidFeat_student)

        # Consistency branch (same teacher input, different path)
        consistency_tmidFeat = dimReduction_student(inputs_tensor_teacher.detach())
        consistency_tPredict = classifier_student(consistency_tmidFeat)

        # ========== LOSS COMPUTATION ==========
        # Distillation loss (weighted by attention)
        loss_c = ce_cri(
            F.log_softmax(tPredict_student, dim=-1),
            F.softmax(tPredict_teacher.detach(), dim=-1),
        )
        loss_c = torch.einsum("ns,n->ns", loss_c, attention_score)
        loss_c = loss_c.mean()

        # Consistency loss (prediction level)
        loss_w1 = ce_cri(
            F.log_softmax(consistency_tPredict, dim=-1),
            F.softmax(tPredict_teacher, dim=-1),
        ).mean()

        # Consistency loss (feature level)
        loss_w2 = ce_cri(
            F.log_softmax(consistency_tmidFeat, dim=-1),
            F.softmax(tmidFeat_teacher, dim=-1),
        ).mean()

        # Total loss
        loss = loss_c + 0.5 * loss_w1 + 0.5 * loss_w2

        # ========== OPTIMIZATION ==========
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            dimReduction_student.parameters(), params.grad_clipping
        )
        torch.nn.utils.clip_grad_norm_(
            classifier_student.parameters(), params.grad_clipping
        )
        torch.nn.utils.clip_grad_norm_(model_student.parameters(), params.grad_clipping)

        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_loss_c += loss_c.item()
        total_loss_w1 += loss_w1.item()
        total_loss_w2 += loss_w2.item()

        if i % 200 == 0:
            print(
                f"  [Epoch {epoch+1}, Iter {i}/{len(trainloader)}] "
                f"Loss: {loss.item():.4f} | "
                f"L_c: {loss_c.item():.4f} | "
                f"L_w1: {loss_w1.item():.4f} | "
                f"L_w2: {loss_w2.item():.4f}"
            )

    # Epoch summary
    avg_loss = total_loss / len(trainloader)
    avg_loss_c = total_loss_c / len(trainloader)
    avg_loss_w1 = total_loss_w1 / len(trainloader)
    avg_loss_w2 = total_loss_w2 / len(trainloader)

    print(f"\n[Epoch {epoch+1} Summary]")
    print(
        f"  Avg Loss: {avg_loss:.4f} | L_c: {avg_loss_c:.4f} | "
        f"L_w1: {avg_loss_w1:.4f} | L_w2: {avg_loss_w2:.4f}"
    )

    # Validation
    scheduler.step()
    print("\nValidation Results on val set:")
    val_mae, val_acc, val_f1, val_auc = TestModel(valloader)

    print("\nValidation Results on test set:")
    test_mae, test_acc, test_f1, test_auc = TestModel(testloader)

    # Save best model
    if val_mae < best_mae:
        best_mae = val_mae
        best_acc = val_acc
        final_test_mae = test_mae
        final_test_acc = test_acc
        save_path = f"OCT_distilled_student_best.pth"
        torch.save(
            {
                "model_student": model_student.state_dict(),
                "dim_reduction": dimReduction_student.state_dict(),
                "classifier": classifier_student.state_dict(),
                "epoch": epoch,
                "mae": val_mae,
                "acc": val_acc,
                "f1": val_f1,
                "auc": val_auc,
            },
            save_path,
        )
        print(
            f"  *** NEW BEST MODEL SAVED: Val MAE={val_mae:.4f}, Val ACC={val_acc:.4f} ***\n"
        )
    else:
        print()

print("\n" + "=" * 50)
print("Training Complete!")
print(f"Final Test MAE: {final_test_mae:.4f} | Final Test ACC: {final_test_acc:.4f}")
print("=" * 50)
