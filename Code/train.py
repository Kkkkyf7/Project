import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from model import InceptionSS

def main():


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    logs = open('logs.txt', 'w')

    batch_size = 70
    epochs = 100


    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(256),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(degrees=20),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((256, 256)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


    data_root = "D:\\model"
    image_path = os.path.join(data_root, "data")  #set path
    print("Data set path:", image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)


    retina_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in retina_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    net = InceptionSS(num_classes=4)
    model_weight_path = "./model.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')
    net.load_state_dict(pre_weights)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = './model.pth'
    train_steps = len(train_loader)

    #Move the loss and accuracy records out of the epoch loop
    loss_records = []
    accuracy_records = []
    f1_scores = []
    precisions = []
    recalls = []


    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # calculate train accuracy
            _, predicted = torch.max(logits, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)
            # calculate train accuracy
        train_accuracy = correct_train / total_train

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        predictions = []
        targets = []
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                #loss = loss_function(outputs, test_labels)
                val_loss = loss_function(outputs, val_labels.to(device))

                #accumulate the loss
                val_loss += loss.item() * val_images.size(0)

                predict_y = torch.max(outputs, dim=1)[1]
                predictions.extend(predict_y.cpu().numpy())
                targets.extend(val_labels.cpu().numpy())

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)
        val_accurate = acc / val_num
        val_loss /= val_num

        # calculate f1 score
        f1_macro = f1_score(targets, predictions, average='macro')
        f1_scores.append(f1_macro)

        #recall score
        recall = recall_score(targets, predictions, average='macro')
        recalls.append(recall)

        #Precision score
        precision = precision_score(targets, predictions, average='macro')
        precisions.append(precision)


        # Record loss and accuracy for each epoch
        loss_records.append(running_loss / len(train_loader))
        accuracy_records.append(val_accurate)

        # Print and log the results
        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f val_loss: %.3f val_accuracy: %.3f  val_F1(Macro): %.3f '
              ' recall_score: %3f  precision_score: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accuracy,  val_loss, val_accurate, f1_macro,
               recall, precision))
        with open('logs1.txt', 'a') as file:
            file.write('[epoch %d / %d] train_loss: %.3f  train_accuracy: %.3f  val_loss: %.3f val_accuracy: %.3f  val_F1(Macro): %.3f '
                       'recall_score: %3f  precision_score: %.3f' %
                       (epoch + 1, epochs, running_loss / train_steps, train_accuracy, val_loss, val_accurate, f1_macro,
                        recall, precision))
            file.write("\n")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(targets, predictions)

    # Calculate TP, FN, FP, TN for each class
    for i in range(4):
        conf_matrix = confusion_matrix(targets, predictions)

        # Calculate TP, FN, FP, TN for each class
        for i in range(4):
            TP = conf_matrix[i, i]
            FN = np.sum(conf_matrix[i, :]) - TP
            FP = np.sum(conf_matrix[:, i]) - TP
            TN = np.sum(conf_matrix) - TP - FN - FP

            print(f"Class {i}:")
            print(f"True Positives (TP): {TP}")
            print(f"False Negatives (FN): {FN}")
            print(f"False Positives (FP): {FP}")
            print(f"True Negatives (TN): {TN}")

    print('Finished Training')


if __name__ == '__main__':
    main()
