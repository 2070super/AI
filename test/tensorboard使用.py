from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("logs")
for i in range(1,100):
    writer.add_scalar("y=x",i,i)
writer.close()