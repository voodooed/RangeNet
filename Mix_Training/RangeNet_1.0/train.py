import torch
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_fn(loader, model, optimizer, loss_fn, scaler):  # train_fn is going to do one epoch of training
    loop = tqdm(loader) # For progress bar
    losses = []
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):

        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast(): #Float 16 training
            predictions = model(data)
            predictions = torch.log(predictions)
        
       
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # calculate running loss
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)

        loop.set_postfix(loss=running_loss)

    return running_loss
    #return sum(losses) / len(losses)

def val_fn(loader, model, loss_fn):  
    loop = tqdm(loader) 
    losses = []

    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)
          

            predictions = model(data)
            predictions = torch.log(predictions)

            loss = loss_fn(predictions, targets)
         
        

            running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)

            loop.set_postfix(loss=running_loss)
    
    model.train()  # Set the model back to training mode

    return running_loss
    #return sum(losses) / len(losses) 



def test_fn(loader, model, loss_fn):
    loop = tqdm(loader)
    losses = []
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)


            predictions = model(data)
            predictions = torch.log(predictions)
            
            loss = loss_fn(predictions, targets)

            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

           

            loop.set_postfix(loss=running_loss)

            

    # No need to switch back to train mode since there will be no more training after testing
    # model.train() 

    return (sum(losses) / len(losses))
