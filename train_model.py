from imports import *

def evaluate_model(net, dataloader,criterion):
    net.to(DEVICE)
    running_error = 0.0
    running_loss = 0.0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels, path = data['image'], data['label'], data['path']
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs,fc2_x = net(inputs)
            _, argmax = torch.max(outputs, 1)
            err = (argmax.squeeze() != labels).float().mean()
            loss = criterion(outputs, labels)
            running_error += err.item()
            running_loss += loss.item()
    return running_loss/(i+1),running_error/(i+1)

def add_images_to_TB(data_loaders,writer):
    data = iter(data_loaders['train']).next()
    images = data['image']
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image(TB_DIR, img_grid)
    writer.close()
    return


###########################################################################################################

def train_model(model, criterion, optimizer,dataloaders,writer, N_iter=50, num_epochs=NUM_EPOCHS):
    #init
    model.to(DEVICE)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    t = tqdm(total=num_epochs*len(dataloaders['train'])) # Initialise tqdm

    features_for_cluster = None #initializing the tensor for all batches 
    path_for_cluster = [] #None #initializing the tensor for all batches

    print('-' * 10)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        running_loss = 0.0
        running_train_err = 0.0

        running_loss_l = []
        running_train_err_l = []
        val_loss_N_iter_l = []
        val_error_N_iter_l = []

        for i, data in enumerate(dataloaders['train'], 0):
            model.train()
            t.update(1)

            inputs, labels, path = data['image'], data['label'], data['path']
            inputs, labels = inputs.to(DEVICE), labels.to(device=DEVICE, dtype=torch.int64)
            optimizer.zero_grad()
    
            #eval and calc loss
            outputs,features_for_cluster_a = model(inputs)
            
            if epoch+1 == num_epochs: #saving the features all batches only of the last epoch for clustering                
                path_for_cluster.append(path)            
                if features_for_cluster is None:
                    features_for_cluster = features_for_cluster_a.detach().clone().cpu()
                else:
                    features_for_cluster = torch.cat((features_for_cluster, features_for_cluster_a.detach().clone().cpu()), dim=0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #calc error
            labels_unsq = torch.unsqueeze(labels,1)

            _, index = torch.sort(outputs,descending=True,dim=0)
            cut_off = 1.0
            precision_it = 0.0
            recall_it = 0.0

            for i in index[:,1].tolist():
                output_new_test = torch.zeros(outputs.size(0),outputs.size(1)+1)
                output_new_test[:,:1]= 0
                output_new_test[:,1:2]= outputs[:,1:2]>=outputs[:,1:2][i]
                output_new_test[:,2:3]= labels_unsq[:,:1]
                n_classified_pos = float(output_new_test[:,1:2].sum())
                if n_classified_pos > 0.0:
                    precision = float((output_new_test[:,1:2] * output_new_test[:,2:3]).sum()/n_classified_pos)
                else:
                    precision = 1.0
                n_true_pos = float(output_new_test[:,2:3].sum())
                if n_true_pos > 0.0:
                    recall = float((output_new_test[:,1:2] * output_new_test[:,2:3]).sum()/n_true_pos)
                else:
                    recall = 1.0

                if precision > 0.9:
                    cut_off = float(outputs[:,1:2][i][0])
                    precision_it = precision
                    recall_it = recall

            arg90 = (outputs[:,1:2]>=cut_off).squeeze()*1
            err = (arg90.squeeze() != labels).float().mean()
            loss = criterion(outputs, labels)
            prec =  precision_it
            rec = recall_it
 
            #update matrices
            running_loss += loss.item()
            running_train_err += err            
            
            if i % N_iter == N_iter-1:    # every 10 mini-batches update evaluation matrices
                TB_step = epoch * len(dataloaders['train']) + i
                #train results
                # ...log the train running loss
                writer.add_scalar('Loss/train',running_loss / N_iter,TB_step)
                writer.add_scalar('Error/train',running_train_err / N_iter,TB_step)

                running_loss_l.append(running_loss / N_iter)
                running_train_err_l.append(running_train_err / N_iter)

                running_loss = 0.0
                running_train_err = 0.0 

                #val results
                val_loss_N_iter,val_error_N_iter = evaluate_model(model, dataloaders['val'],criterion)
                writer.add_scalar(f'Loss/val', val_loss_N_iter, TB_step)
                writer.add_scalar(f'Error/val', val_error_N_iter, TB_step)

                val_loss_N_iter_l.append(val_loss_N_iter)
                val_error_N_iter_l.append(val_error_N_iter)

        _,val_err = evaluate_model(model, dataloaders['val'],criterion)
        val_acc = 1-val_err
        print(f'Val Acc: {np.round(val_acc,4)}')
        print('-' * 10)

        # deep copy the model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best val Acc: {np.round(best_acc,4)}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), TB_DIR+'/model.pt')

    return model,features_for_cluster,path_for_cluster