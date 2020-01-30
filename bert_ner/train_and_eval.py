""" Train and Eval Module """

from transformers import AdamW
from tqdm import tqdm, trange
import math
import torch
import torch.nn.functional as F
from seqeval.metrics import (classification_report,
                             accuracy_score, f1_score)


def train_model(
    model,
    train_dataloader,
    tr_inputs,
    epochs=1,
    batch_num=32,
    max_grad_norm=1.0,
    FULL_FINETUNING=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    # Cacluate train optimiazaion num
    num_train_optimization_steps = int(
        math.ceil(len(tr_inputs) / batch_num) / 1
    ) * epochs

    # True: fine tuning all the layers
    # False: only fine tuning the classifier layers

    if FULL_FINETUNING:
        # Fine tune model all layer parameters
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
                'weight_decay_rate': 0.01},
            {'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
                'weight_decay_rate': 0.0}
        ]
    else:
        # Only fine tune classifier parameters
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

    # TRAIN loop
    model.train()

    print("***** Running training *****")
    print("  Num examples = %d" % (len(tr_inputs)))
    print("  Batch size = %d" % (batch_num))
    print("  Num steps = %d" % (num_train_optimization_steps))
    for _ in trange(epochs, desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        pbar = tqdm(total=num_train_optimization_steps)
        for step, batch in enumerate(train_dataloader):
            pbar.update(1)
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # forward pass
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            loss, scores = outputs[:2]
            if n_gpu > 1:
                # When multi gpu, average it
                loss = loss.mean()

            # backward pass
            loss.backward()

            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm)

            # update parameters
            optimizer.step()
            optimizer.zero_grad()

        # print train loss per epoch
        pbar.close()
        print("Train loss: {}".format(tr_loss/nb_tr_steps))


def eval_model(
    model,
    valid_dataloader,
    val_inputs,
    idx2name_dic,
    output_file_path,
    batch_num=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []

    print("***** Running evaluation *****")
    print("  Num examples ={}".format(len(val_inputs)))
    print("  Batch size = {}".format(batch_num))
    for step, batch in enumerate(valid_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch

    #     if step > 2:
    #         break

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None,
                            attention_mask=input_mask,)
            # For eval mode, the first result of outputs is logits
            logits = outputs[0]

        # Get NER predict result
        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()

        # Get NER true result
        label_ids = label_ids.to('cpu').numpy()

        # Only predict the real word, mark=0, will not calculate
        input_mask = input_mask.to('cpu').numpy()

        # Compare the valuable predict result
        for i, mask in enumerate(input_mask):
            # Real one
            temp_1 = []
            # Predict one
            temp_2 = []

            for j, m in enumerate(mask):
                # Mark=0, meaning its a pad word, dont compare
                if m:
                    # Exclude the X label
                    if idx2name_dic[label_ids[i][j]] != "X" and idx2name_dic[label_ids[i][j]] != "[CLS]" and idx2name_dic[label_ids[i][j]] != "[SEP]":
                        temp_1.append(idx2name_dic[label_ids[i][j]])
                        temp_2.append(idx2name_dic[logits[i][j]])
                else:
                    break

            y_true.append(temp_1)
            y_pred.append(temp_2)

    print("f1 socre: %f" % (f1_score(y_true, y_pred)))
    print("Accuracy score: %f" % (accuracy_score(y_true, y_pred)))

    # Get acc , recall, F1 result report
    report = classification_report(y_true, y_pred, digits=4)

    # Save the report into file
    with open(output_file_path, "w") as writer:
        print("***** Eval results *****")
        print("\n%s" % (report))
        print("f1 socre: %f" % (f1_score(y_true, y_pred)))
        print("Accuracy score: %f" % (accuracy_score(y_true, y_pred)))

        writer.write("f1 socre:\n")
        writer.write(str(f1_score(y_true, y_pred)))
        writer.write("\n\nAccuracy score:\n")
        writer.write(str(accuracy_score(y_true, y_pred)))
        writer.write("\n\n")

        writer.write(report)
