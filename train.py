def val_per_epoch(self, epoch):
    self.model.eval()

    losses = AverageMeter()

    total_correct = 0
    total_samples = 0

    TP = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for i, (x, label, filename, info, human) in enumerate(self.val_loader):

            if self.enable_cuda:
                x = x.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            pred = self.model(x)
            loss = self.criterion(pred, label)

            bsz = label.size(0)
            losses.update(loss.item(), bsz)

            pred_label = pred.argmax(dim=1)

            total_correct += (pred_label == label).sum().item()
            total_samples += bsz

            TP += ((pred_label == 1) & (label == 1)).sum().item()
            FP += ((pred_label == 1) & (label == 0)).sum().item()
            FN += ((pred_label == 0) & (label == 1)).sum().item()

    # ✅ epoch 结束统一算
    acc = total_correct / total_samples
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    # 更新最优判断用变量
    self.temp_acc = acc
    self.temp_pre = precision
    self.temp_re = recall

    record = f"{epoch} acc:{acc} recall:{recall} precision:{precision}\n"
    self.file_name_test.writelines(record)
