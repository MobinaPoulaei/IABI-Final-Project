from utils.network import Classifier_1fc, DimReduction
from utils.Attention import Attention_Gated as Attention
from utils.Attention import Attention_with_Classifier
from utils.OCTEmbededFeatsDataset import OCTEmbededFeatsDataset
from utils.RandMixup import randmixup
import argparse
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np

parser = argparse.ArgumentParser(description='OCT ICMIL Training')

parser.add_argument('--name', default='oct_icmil', type=str)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--droprate', default=0, type=float)
parser.add_argument('--droprate_2', default=0, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_cls', default=2, type=int,
                    help='2 for binary (NORMAL vs DISEASE), 4 for multi-class')
parser.add_argument('--data_path', default='/kaggle/working/Retinal', type=str)
parser.add_argument('--numGroup', default=5, type=int)
parser.add_argument('--mDim', default=512, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--numLayer_Res', default=0, type=int)
params = parser.parse_args()

# ==================== MODELS ====================
classifier  = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
attention   = Attention(params.mDim).to(params.device)
dimReduction = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
attCls      = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls,
                                        droprate=params.droprate_2).to(params.device)

# ==================== DATASETS ====================
trainset = OCTEmbededFeatsDataset(params.data_path, mode='train', level=0, num_cls=params.num_cls)
valset   = OCTEmbededFeatsDataset(params.data_path, mode='val',   level=0, num_cls=params.num_cls)
testset  = OCTEmbededFeatsDataset(params.data_path, mode='test',  level=0, num_cls=params.num_cls)


def collate_features(batch):
    """Custom collate function for variable-length bags"""
    img    = [torch.from_numpy(item[0]).to(params.device) for item in batch]
    labels = [torch.tensor(item[1]).to(params.device) for item in batch]
    return img, labels


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True,
                                          drop_last=False, collate_fn=collate_features)
valloader   = torch.utils.data.DataLoader(valset,   batch_size=1, shuffle=False, drop_last=False)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=1, shuffle=False, drop_last=False)

# ==================== OPTIMIZERS ====================
trainable_parameters  = list(classifier.parameters())
trainable_parameters += list(attention.parameters())
trainable_parameters += list(dimReduction.parameters())

optimizer0 = torch.optim.Adam(trainable_parameters,   lr=params.lr, weight_decay=params.weight_decay)
optimizer1 = torch.optim.Adam(attCls.parameters(),    lr=params.lr, weight_decay=params.weight_decay)

# ==================== LOSS ====================
if params.num_cls == 2:
    # Upweight disease class slightly to compensate for residual imbalance
    class_weights = torch.tensor([1.0, 1.5]).to(params.device)
    ce_cri = torch.nn.CrossEntropyLoss(reduction='none', weight=class_weights).to(params.device)
else:
    ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)

best_auc   = 0
best_epoch = -1
test_auc   = 0


# ==================== METRICS ====================
def compute_metrics(y_true, y_score, num_cls):
    y_pred = np.argmax(y_score, axis=1)
    acc    = np.mean(y_true == y_pred)

    if num_cls == 2:
        try:
            auc = roc_auc_score(y_true, y_score[:, 1])
        except Exception:
            auc = 0.5
        f1        = f1_score(y_true, y_pred, average='binary', pos_label=1)
        precision = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        recall    = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    else:
        try:
            auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.5
        f1        = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)

    return auc, acc, f1, precision, recall

# ==================== TEST / VAL FUNCTION ====================
def TestModel(test_loader):
    classifier.eval()
    dimReduction.eval()
    attention.eval()
    attCls.eval()

    y_score = []
    y_true  = []

    for i, data in enumerate(test_loader):
        inputs, labels = data

        labels = labels.data.numpy().tolist()
        slide_pseudo_feat = []
        inputs_pseudo_bags = torch.chunk(inputs.squeeze(0), params.numGroup, dim=0)

        for subFeat_tensor in inputs_pseudo_bags:
            subFeat_tensor = subFeat_tensor.to(params.device)
            with torch.no_grad():
                tmidFeat = dimReduction(subFeat_tensor)
                tAA      = attention(tmidFeat).squeeze(0)
            tattFeats        = torch.einsum('ns,n->ns', tmidFeat, tAA)
            tattFeat_tensor  = torch.sum(tattFeats, dim=0).unsqueeze(0)
            slide_pseudo_feat.append(tattFeat_tensor)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        gSlidePred        = torch.softmax(attCls(slide_pseudo_feat), dim=1)

        pred = gSlidePred.mean(dim=0).cpu().data.numpy()
        y_score.append(pred)
        y_true.append(labels[0])

    y_score = np.array(y_score)
    y_true  = np.array(y_true)

    auc, acc, f1, precision, recall = compute_metrics(y_true, y_score, params.num_cls)
    print('result: auc:{:.4f}, acc:{:.4f}, f1:{:.4f}, precision:{:.4f}, recall:{:.4f}'.format(
        auc, acc, f1, precision, recall))
    return auc, acc, f1, precision, recall


# ==================== INITIAL TEST ====================
print("\n=== Initial Test (before training) ===")
TestModel(testloader)

# ==================== TRAINING LOOP ====================
for ii in range(params.EPOCH):

    for param_group in optimizer1.param_groups:
        curLR = param_group['lr']
        print('\n=== Epoch {}/{} - Learning rate: {} ==='.format(ii + 1, params.EPOCH, curLR))

    classifier.train()
    dimReduction.train()
    attention.train()
    attCls.train()

    epoch_loss_1 = 0
    epoch_loss1  = 0
    num_batches  = 0

    for i, data in enumerate(trainloader):
        inputs, labels = data

        labels = torch.stack(labels)

        mix_inputs, labels_a, labels_b, lmbdas = randmixup(inputs, labels)

        for j in range(len(mix_inputs)):
            inputs  = mix_inputs[j].unsqueeze(0)
            label_a = labels_a[j].unsqueeze(0).to(params.device)
            label_b = labels_b[j].unsqueeze(0).to(params.device)
            lam     = lmbdas[j]

            slide_sub_preds    = []
            slide_sub_labels_a = []
            slide_sub_labels_b = []
            slide_pseudo_feat  = []

            inputs_pseudo_bags = torch.chunk(inputs.squeeze(0), params.numGroup, dim=0)

            for subFeat_tensor in inputs_pseudo_bags:
                slide_sub_labels_a.append(label_a)
                slide_sub_labels_b.append(label_b)
                subFeat_tensor  = subFeat_tensor.to(params.device)
                tmidFeat        = dimReduction(subFeat_tensor)
                tAA             = attention(tmidFeat).squeeze(0)
                tattFeats       = torch.einsum('ns,n->ns', tmidFeat, tAA)
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)
                tPredict        = classifier(tattFeat_tensor)
                slide_sub_preds.append(tPredict)
                slide_pseudo_feat.append(tattFeat_tensor)

            slide_pseudo_feat  = torch.cat(slide_pseudo_feat,  dim=0)
            slide_sub_preds    = torch.cat(slide_sub_preds,    dim=0)
            slide_sub_labels_a = torch.cat(slide_sub_labels_a, dim=0)
            slide_sub_labels_b = torch.cat(slide_sub_labels_b, dim=0)

            # First-tier loss
            loss_1 = (lam       * ce_cri(slide_sub_preds, slide_sub_labels_a).mean() +
                      (1 - lam) * ce_cri(slide_sub_preds, slide_sub_labels_b).mean())

            optimizer0.zero_grad()
            loss_1.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(attention.parameters(),    params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(),   params.grad_clipping)
            optimizer0.step()

            # Second-tier loss
            gSlidePred = attCls(slide_pseudo_feat.detach())
            loss1      = (lam       * ce_cri(gSlidePred, label_a).mean() +
                          (1 - lam) * ce_cri(gSlidePred, label_b).mean())

            optimizer1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(attCls.parameters(), params.grad_clipping)
            optimizer1.step()

            epoch_loss_1 += loss_1.item()
            epoch_loss1  += loss1.item()
            num_batches  += 1

    avg_loss_1 = epoch_loss_1 / num_batches
    avg_loss1  = epoch_loss1  / num_batches
    print('\n[EPOCH {} Summary] avg_loss_1:{:.4f}; avg_loss1:{:.4f}'.format(
        ii + 1, avg_loss_1, avg_loss1))

    # Validation
    print('\n=== Validating on Val Set ===')
    auc, acc, f1, precision, recall = TestModel(valloader)

    if auc > best_auc:
        best_auc   = auc
        best_epoch = ii
        print('\n*** New best AUC: {:.4f} at epoch {}. Testing on Test Set... ***'.format(
            best_auc, best_epoch + 1))
        test_auc, test_acc, test_f1, test_precision, test_recall = TestModel(testloader)  # ← distinct names

        tsave_dict = {
            'classifier':     classifier.state_dict(),
            'dim_reduction':  dimReduction.state_dict(),
            'attention':      attention.state_dict(),
            'att_classifier': attCls.state_dict(),
            'epoch':          ii,
            'num_cls':        params.num_cls,
            'val_auc':        auc,            # ← val auc (from valloader above)
            'test_auc':       test_auc,       # ← now correctly set
            'test_precision': test_precision, # ← now correctly set
            'test_recall':    test_recall,    # ← now correctly set
        }
        print("Saving best model to model_best_oct.pth")
        torch.save(tsave_dict, 'model_best_oct.pth')

print("\n" + "=" * 60)
print("Training Complete!")
print(f"Best Val AUC : {best_auc:.4f} at Epoch {best_epoch + 1}")
print("=" * 60)