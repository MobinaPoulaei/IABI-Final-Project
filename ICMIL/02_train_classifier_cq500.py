from model.network import Classifier_1fc, DimReduction
from model.Attention import Attention_Gated as Attention
from model.Attention import Attention_with_Classifier
import argparse
import torch
from utils.CQ500EmbededFeatsDataset import CQ500EmbededFeatsDataset
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import numpy as np
from RandMixup import randmixup

parser = argparse.ArgumentParser(description="CQ500 ICMIL Training")

parser.add_argument("--name", default="cq500_icmil", type=str)
parser.add_argument("--EPOCH", default=200, type=int)
parser.add_argument("--epoch_step", default="[100]", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--droprate", default="0", type=float)
parser.add_argument("--droprate_2", default="0", type=float)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--lr_decay_ratio", default=0.2, type=float)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--num_cls", default=2, type=int)
parser.add_argument(
    "--data_path", default="/kaggle/working/CQ500_ICH_VS_NORMAL", type=str
)  # Path to .npy features
parser.add_argument("--numGroup", default=5, type=int)
parser.add_argument("--mDim", default=512, type=int)
parser.add_argument("--grad_clipping", default=5, type=float)
parser.add_argument("--numLayer_Res", default=0, type=int)
params = parser.parse_args()

# Models
classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(
    params.device
)
attention = Attention(params.mDim).to(params.device)
dimReduction = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(
    params.device
)
attCls = Attention_with_Classifier(
    L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2
).to(params.device)

# Optional: Load pretrained weights
# pretrained_weights = torch.load('/path/to/model_best.pth')
# classifier.load_state_dict(pretrained_weights['classifier'])
# dimReduction.load_state_dict(pretrained_weights['dim_reduction'])
# attention.load_state_dict(pretrained_weights['attention'])
# attCls.load_state_dict(pretrained_weights['att_classifier'])

# Datasets
trainset = CQ500EmbededFeatsDataset(params.data_path, mode="train", level=0)
valset = CQ500EmbededFeatsDataset(params.data_path, mode="val", level=0)
testset = CQ500EmbededFeatsDataset(params.data_path, mode="test", level=0)


def collate_features(batch):
    """Custom collate function for variable-length bags"""
    img = [torch.from_numpy(item[0]).to(params.device) for item in batch]
    labels = [torch.tensor(item[1]).to(params.device) for item in batch]
    return img, labels


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, drop_last=False, collate_fn=collate_features
)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=1, shuffle=False, drop_last=False
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, drop_last=False
)

# Set models to train mode
classifier.train()
dimReduction.train()
attention.train()
attCls.train()

# Optimizers
trainable_parameters = []
trainable_parameters += list(classifier.parameters())
trainable_parameters += list(attention.parameters())
trainable_parameters += list(dimReduction.parameters())

optimizer0 = torch.optim.Adam(
    trainable_parameters, lr=params.lr, weight_decay=params.weight_decay
)
optimizer1 = torch.optim.Adam(
    attCls.parameters(), lr=params.lr, weight_decay=params.weight_decay
)

best_auc = 0
best_epoch = -1
test_auc = 0

ce_cri = torch.nn.CrossEntropyLoss(reduction="none").to(params.device)


def optimal_thresh(fpr, tpr, thresholds, p=0):
    """Find optimal threshold for binary classification"""
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def TestModel(test_loader):
    """Validation/Test function"""
    classifier.eval()
    dimReduction.eval()
    attention.eval()
    attCls.eval()

    y_score = []
    y_true = []

    for i, data in enumerate(test_loader):
        inputs, labels = data

        labels = labels.data.numpy().tolist()
        slide_pseudo_feat = []
        inputs_pseudo_bags = torch.chunk(inputs.squeeze(0), params.numGroup, dim=0)

        for subFeat_tensor in inputs_pseudo_bags:
            subFeat_tensor = subFeat_tensor.to(params.device)
            with torch.no_grad():
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum("ns,n->ns", tmidFeat, tAA)  # n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  # 1 x fs
            slide_pseudo_feat.append(tattFeat_tensor)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        gSlidePred = torch.softmax(attCls(slide_pseudo_feat), dim=1)

        pred = gSlidePred.mean(dim=0)
        pred = pred.cpu().data.numpy()
        y_score.append(pred)
        y_true.append(labels[0])

    y_score = np.array(y_score)
    y_true = np.array(y_true)

    print("y_score shape:", y_score.shape)
    print("y_true shape:", y_true.shape)

    y_pred = np.argmax(y_score, axis=1)
    y_prob = y_score[:, 1]

    # Metrics for multi-class
    acc = np.mean(y_true == y_pred)

    # AUC (one-vs-rest)
    try:
        # auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    # F1 score (macro average)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(
        "validation results\nauc: {:.4f}, acc: {:.4f}, f1: {:.4f}".format(auc, acc, f1)
    )

    return auc, acc, f1


# Initial test (optional)
print("\n=== Initial Test (before training) ===")
TestModel(testloader)

# Training loop
for ii in range(params.EPOCH):

    for param_group in optimizer1.param_groups:
        curLR = param_group["lr"]
        print(
            "\n=== Epoch {}/{} - Learning rate: {} ===".format(
                ii + 1, params.EPOCH, curLR
            )
        )

    classifier.train()
    dimReduction.train()
    attention.train()
    attCls.train()

    epoch_loss_1 = 0
    epoch_loss1 = 0
    num_batches = 0

    for i, data in enumerate(trainloader):
        inputs, labels = data

        # Convert labels list to tensor
        labels = torch.stack(labels)

        # Apply mixup augmentation
        mix_inputs, labels_a, labels_b, lmbdas = randmixup(inputs, labels)

        for j in range(len(mix_inputs)):
            inputs = mix_inputs[j].unsqueeze(0)
            label_a = labels_a[j].unsqueeze(0)
            label_b = labels_b[j].unsqueeze(0)
            lam = lmbdas[j]

            label_a = label_a.to(params.device)
            label_b = label_b.to(params.device)

            slide_sub_preds = []
            slide_sub_labels_a = []
            slide_sub_labels_b = []
            slide_pseudo_feat = []
            inputs_pseudo_bags = torch.chunk(inputs.squeeze(0), params.numGroup, dim=0)

            for subFeat_tensor in inputs_pseudo_bags:
                slide_sub_labels_a.append(label_a)
                slide_sub_labels_b.append(label_b)
                subFeat_tensor = subFeat_tensor.to(params.device)
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
                tattFeats = torch.einsum("ns,n->ns", tmidFeat, tAA)  # n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  # 1 x fs
                tPredict = classifier(tattFeat_tensor)  # 1 x num_cls
                slide_sub_preds.append(tPredict)
                slide_pseudo_feat.append(tattFeat_tensor)

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  # numGroup x num_cls
            slide_sub_labels_a = torch.cat(slide_sub_labels_a, dim=0)  # numGroup
            slide_sub_labels_b = torch.cat(slide_sub_labels_b, dim=0)  # numGroup

            # Mixup loss
            loss_1 = (
                lam * ce_cri(slide_sub_preds, slide_sub_labels_a).mean()
                + (1 - lam) * ce_cri(slide_sub_preds, slide_sub_labels_b).mean()
            )

            optimizer0.zero_grad()
            loss_1.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                dimReduction.parameters(), params.grad_clipping
            )
            torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(
                classifier.parameters(), params.grad_clipping
            )
            optimizer0.step()

            # Optimization for the second tier
            gSlidePred = attCls(slide_pseudo_feat.detach())
            loss1 = (
                lam * ce_cri(gSlidePred, label_a).mean()
                + (1 - lam) * ce_cri(gSlidePred, label_b).mean()
            )

            optimizer1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(attCls.parameters(), params.grad_clipping)
            optimizer1.step()

            epoch_loss_1 += loss_1.item()
            epoch_loss1 += loss1.item()
            num_batches += 1

    # Epoch summary
    avg_loss_1 = epoch_loss_1 / num_batches
    avg_loss1 = epoch_loss1 / num_batches
    print(
        "\n[EPOCH {} Summary] avg_loss_1:{:.4f}; avg_loss1:{:.4f}".format(
            ii + 1, avg_loss_1, avg_loss1
        )
    )

    # Validation
    print("\n=== Validating on Val Set ===")
    auc, acc, f1 = TestModel(valloader)

    if auc > best_auc:
        best_auc = auc
        best_epoch = ii
        print(
            "\n*** New best AUC: {:.4f} at epoch {}. Testing on Test Set... ***".format(
                best_auc, best_epoch + 1
            )
        )
        test_auc, test_acc, test_f1 = TestModel(testloader)

        tsave_dict = {
            "classifier": classifier.state_dict(),
            "dim_reduction": dimReduction.state_dict(),
            "attention": attention.state_dict(),
            "att_classifier": attCls.state_dict(),
            "epoch": ii,
            "val_auc": auc,
            "test_auc": test_auc,
        }
        print("Saving best model to model_best_cq500.pth")
        torch.save(tsave_dict, "model_best_cq500.pth")

print("\n" + "=" * 60)
print("Training Complete!")
print(f"Best Val AUC: {best_auc:.4f} at Epoch {best_epoch+1}")
print("=" * 60)
