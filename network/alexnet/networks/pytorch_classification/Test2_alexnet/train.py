import os
import json

import torch
from torch import tensor
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from visdom import Visdom

from model import AlexNet


def main():
    # viz = Visdom()
    # viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()
                                     ]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor()
                                   ])}

    data_root = "/home/zhongsy/datasets/dataset/"  # get data root path
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"),
                                         transform=data_transform["train"])

    # print(train_dataset.imgs)
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 1
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=2, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 30
    save_path = './AlexNet.pt'
    best_acc = 0.0
    train_steps = len(train_loader)
    global_step = 0
    for epoch in range(epochs):
        # train
        epochloss = 100000
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data

            # print("label: ", labels, labels.dtype)
            optimizer.zero_grad()
            outputs = net(images.to(device))
            # print("imges: ", images, images.dtype)
            # outputs_ = outputs.squeeze()
            # print("output__ : ", outputs_)
            # outputs_ = outputs.to(torch.float)
            loss = loss_function(outputs, labels.to(device))
            # loss = loss.to(torch.float)
            if epochloss > loss:
                epochloss = loss
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # viz.line([epochloss.cpu().detach().numpy()], [global_step],
            #  win='train_loss', update='append')
        global_step += 1

        print("[ start val ]")
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_labels.unsqueeze(1)
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                # print("prect ;", predict_y)
                # outputs = outputs.squeeze()
                # print("out_puts: ", outputs)
                # a = torch.gt(outputs, 0.5)
                # print("a ", a)
                # for i, (data, label_) in enumerate(zip(outputs, val_labels)):
                #     if abs(data-label_) <= 0.5:
                #         acc += 1
                # viz.images(val_images.view(-1, 3, 224, 224), win='x')
                # viz.text(str(predict_y.detach().cpu().numpy()),
                #  win='pred', opts=dict(title='pred'))
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net, save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
