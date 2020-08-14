
# Valid is labeled as 1 i.e. Positive and invalid is 0 i.e. Negative

from imports import *
    

def eval_model(model,data_loaders):
    '''
    eval_model methods evaluate model confusion matrix with respect to the binary classification.
    It prints model performance report and saves model error in TB_DIR/gap_analysis.csv file
    Input params:
    ~~~~~~~~~~~~
    -- model : pretrained model
    -- data_loaders : the dataset to evaluate, as mentioned the data_loaders['val'] loader of MyDataset()
    Return:
    ~~~~~~
    -- pd DataFrame which includes model errors
    '''
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    
    t = tqdm(total=len(data_loaders)) # Initialise tqdm
    
    gaped_examples = []
    features_for_cluster_eval = None #initializing the tensor for all batches 
    path_for_cluster_eval = [] #initializing the tensor for all batches
    
    true_label_list = None #initializing the tensor for all batches 
    predicted_prob_list = None #initializing the tensor for all batches 
    
    predicted_label_for_cluster = None #initializing the tensor for all batches
    
    model.eval() #enter eval mode
    with torch.no_grad():
        for i, data in enumerate(data_loaders, 0):
            t.update(1)
            inputs, labels, path = data['image'], data['label'], data['path']
            inputs, labels = inputs.to(DEVICE), labels.to(device=DEVICE, dtype=torch.int64)
            
            outputs,features_for_cluster_a = model(inputs)

            #saving the features of all batches for clustering                
            if features_for_cluster_eval is None:
                features_for_cluster_eval = features_for_cluster_a.detach().clone().cpu()
            else:
                features_for_cluster_eval = torch.cat((features_for_cluster_eval, features_for_cluster_a.detach().clone().cpu()), dim=0)
            path_for_cluster_eval.append(path)
            
            prob , predicted = torch.max(outputs, 1)

            if true_label_list is None:
                true_label_list = labels.detach().clone().cpu()
            else:
                true_label_list = torch.cat((true_label_list, labels.detach().clone().cpu()), dim=0)
            
            if predicted_prob_list is None:
                predicted_prob_list = outputs[:,1].detach().clone().cpu()
            else:
                predicted_prob_list = torch.cat((predicted_prob_list, outputs[:,1].detach().clone().cpu()), dim=0)

            if predicted_label_for_cluster is None:
                predicted_label_for_cluster = predicted.detach().clone().cpu()
            else:
                predicted_label_for_cluster = torch.cat((predicted_label_for_cluster, predicted.detach().clone().cpu()), dim=0)
                
                
            for j in range(len(labels)):
                true_label = labels[j].item()
                pred_label = predicted[j].item()            
                    
                if (true_label==1):
                    if (pred_label==1):
                        TP+=1
                    else: 
                        FN+=1
                        example = {'type':'FN', 'path':path[j],
                                   'true':CLASSES[true_label],
                                   'pred':CLASSES[pred_label],
                                   'prob' : prob[j].item()}
                        gaped_examples.append(example)
                else:
                    if (pred_label==1):
                        FP+=1
                        example = {'type':'FP', 'path':path[j],
                                   'true':CLASSES[true_label],
                                   'pred':CLASSES[pred_label],
                                   'prob' : prob[j].item()}
                        gaped_examples.append(example)
                    else:
                        TN+=1

    print(f'Support:   {TP+TN+FP+FN}')
    print(f'Accuracy:  {np.round(100*(TP+TN)/(TP+TN+FP+FN),3)}%')
    print(f'Recall:    {np.round(100*(TP/(TP+FN)),3)}%')
    print(f'Precision: {np.round(100*(TP/(TP+FP)),3)}%')
    print(f'F1:        {np.round(100*(2*((TP/(TP+FP))*(TP/(TP+FN)))/((TP/(TP+FP))+(TP/(TP+FN)))),3)}%')
    
    lr_precision, lr_recall, thr = precision_recall_curve(np.asarray(true_label_list), np.exp(np.asarray(predicted_prob_list)))
    data_2_plot = go.Scattergl(x = lr_recall, y = lr_precision, mode='lines', text=thr)
    layout = go.Layout(title='Precision-recall',xaxis=dict(title='Recall'),yaxis=dict(title='Precision'))
    fig = go.Figure(data=data_2_plot,layout=layout)
    py.iplot(fig)

    gaped_examples = pd.DataFrame(gaped_examples)
    gaped_examples.to_csv(f'{TB_DIR}/gap_analysis.csv')
    return gaped_examples,features_for_cluster_eval,path_for_cluster_eval,predicted_label_for_cluster
