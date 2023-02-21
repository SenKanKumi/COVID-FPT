import torch
import os
from tqdm import tqdm
from build.DataLoader import Creat_DataSet
from build.Optimizer import SelectOptim, SelectScheduler
from Models.MyModel import FPN
from torch.nn import functional as F
from utils import CreateLog, Evaluate, Confusion_matrix, ServiceInform


def Train_FPSIT(size="tiny", data="base", batch=32, opt="adamw", base_lr=1e-5, train_lr=1e-3, sche="exponent", ans=5):
    r"""Train the Channel Attention Transformer-CAEmbed-(CAT2)

    Args concludes [size, data, batch, opt, base_lr, train_lr, sche, ans]
        size[str]: Set the size of Swin Transformer Model, such as "tiny","small","base", default="tiny"
        data[str]: Select which data will be used, such as "base","small","tiny", default="base"
        batch[int]: Set the number of dataloader load for each epoch, default=32
        opt[str]: Set the optimizer of net, such as "adam","adamw", default="adamw"
        base_lr[float]: Set the lr of Baseline Net which is pretrained, default=1e-5
        train_lr[float]: Set the lr of new Net which is own, default=1e-3
        sche[str]: Set the scheduler of net, such as "exponent","cosine","linear", default="exponent"
        ans[int]: Set the times for Training, default=5
    """

    # build the CAT2 model
    model = FPN(load=True)
    device = torch.device("cuda:0")
    model = model.to(device)

    # build the dataloader
    Train_Dataloader = Creat_DataSet(data=data, mode="train", batch=batch, shuffle=True)
    Train_size = len(Train_Dataloader)
    Val_Dataloader = Creat_DataSet(data=data, mode="val", batch=batch, shuffle=True)
    Val_size = len(Val_Dataloader)

    # build the optimizer and set the different learning rate
    TrainParams = model.R_FPSIT.parameters()
    BaseParam = filter(lambda p: id(p) not in list(map(id, TrainParams)), model.parameters())
    params = [
        {"params": BaseParam, "lr": base_lr},
        {"params": TrainParams}
    ]
    optimizer = SelectOptim(params, opt, lr=train_lr)
    # set the scheduler
    scheduler = SelectScheduler(optimizer, sche)

    # set loss function
    loss = torch.nn.CrossEntropyLoss()

    # others setting
    MAccuracy, T_iter = 0, 0
    file = CreateLog(epoch=100, batch=batch,
                     model="FPSIT_{} on {} dataset".format(size, data),
                     optim="{} {} {}".format(opt, base_lr, train_lr),
                     sche="{} 5".format(sche),
                     info="")

    # start training
    for epoch in range(100):
        print("\nNow Epoch : {}".format(epoch + 1))
        trainBar = tqdm(enumerate(Train_Dataloader), total=len(Train_Dataloader), desc="Train ", mininterval=2)
        model.train()
        train_loss = 0
        for index, (x, y) in trainBar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            local_loss = loss(out, y)
            train_loss += local_loss.item()
            local_loss.backward()
            optimizer.step()

        # start evalidating
        ValBar = tqdm(enumerate(Val_Dataloader), total=len(Val_Dataloader), desc="Val", mininterval=2)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            matrix = torch.zeros(3, 3)
            for index, (x, y) in ValBar:
                x, y = x.to(device), y.to(device)
                out = model(x)
                local_loss = loss(out, y)
                val_loss += local_loss.item()
                out = F.log_softmax(out, dim=1).data
                pred = out.data.max(1, keepdim=True)[1].squeeze()
                matrix = Confusion_matrix(pred, y, matrix)
            acc, pre, rec = Evaluate(epoch + 1, matrix, train_loss / Train_size, val_loss / Val_size, file)
            # print(model.R_FPSIT.v1, model.R_FPSIT.v2, model.R_FPSIT.v3, model.R_FPSIT.v4)

        # renre learning rate
        scheduler.step()

        # save the model
        T_iter += 1
        if acc >= MAccuracy:
            T_iter = 0
            MAccuracy = max(acc, MAccuracy)
            checkpoint = {"model": model.state_dict()}
            path_checkpoint = "./checkpoint/{}_{}_{}_FPSIT_{}.pth".format(ans + 1, epoch + 1, round(acc, 4), size)
            torch.save(checkpoint, path_checkpoint)

        # if the MAccuracy has stopped to change for 20 epochs, stop training
        # if T_iter == 20 :
        #     print("----Model training has stopped----")
        #     break


if __name__ == "__main__":
    for ans in range(1):
        # ServiceInform("Starting the {} times Training".format(ans + 1))
        Train_FPSIT(size="tiny", data="base", batch=64, opt="adamw", base_lr=2e-5, train_lr=2e-4, sche="warmup",
                    ans=ans)
        torch.cuda.empty_cache()
