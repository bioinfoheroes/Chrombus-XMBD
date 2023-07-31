def train(loader):
    model.train()
    running_loss = 0
    for data in loader: # Iterate in batches over the training dataset.
        data = data.to(device)
        z = model.encode(data.x, data.edge_index, data.batch) #encoding output        
        pred = model.decoder(z, data.edge_index, sigmoid=False) 
        loss = criterion(pred, data.edge_attr.view(-1)) # Compute the loss.
        loss.backward() # Derive gradients.  
        optimizer.step() # Update parameters based on gradients.
        optimizer.zero_grad() # Clear gradients.
        running_loss += loss
    #print(f'total loss: {(running_loss/len(loader)):4f}')

def test(loader):
    model.eval()
    mse = 0
    pred_cor = 0
    for data in loader: # Iterate in batches over the training/test dataset.
        data = data.to(device)
        z = model.encode(data.x, data.edge_index, data.batch) 
        pred = model.decoder(z, data.edge_index, sigmoid=False) 
        loss = criterion(pred, data.edge_attr.view(-1)) # Check against ground-truth labels.
        mse += loss.cpu().detach()
        df = pandas.DataFrame({'hic':data.edge_attr[:,0].cpu(), 'pred':pred.cpu().detach()})
        pred_cor += df.corr().iloc[0,1]
    return mse / len(loader), pred_cor / len(loader) # Derive ratio of correct predictions.

def get_5_fold_dataset(k, train_dataset):
    k_n = int(0.2 *len(train_dataset))
    if k == 1:
        vali_dat = train_dataset[:k_n]
        train_dat = train_dataset[k_n:]
    elif k == 5:
        vali_dat = train_dataset[-k_n:]
        train_dat = train_dataset[:-k_n]
    else:
        vali_dat = train_dataset[k_n *(k-1):k_n*k]
        train_dat = train_dataset[:k_n*(k-1)] + train_dataset[k_n*k:]
    return([train_dat, vali_dat])
  
###########
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

datdir = './data/'
cell = 'GM12878'
train_datdir = './data/'
dataset = HiCDataset(root='./', name='data', n_seg=128, dataset = 'train', N_chr = 200)
torch.save(dataset,train_datdir + '/non_overlap_epi_traindata.pt')
train_pos = numpy.load(datdir + '/processed/train_pos_list.npy')
numpy.save(train_datdir + '/non_overlap_pos_list.npy',train_pos)
# shutil.rmtree(datdir + '/processed/')
# 随机交换顺序
# 测试集
test_size = 50
train_size = len(dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
torch.save([train_dataset, test_dataset], train_datdir + 'non_overlap_5Fold_dataset.pt')

for N in range(1,6):
    train_dat, vali_dat = get_5_fold_dataset(N, train_dataset)
    train_loader = DataLoader(train_dat, batch_size=16, shuffle=False)
    vali_loader = DataLoader(vali_dat, batch_size=16, shuffle=False)
    # model
    cis_span = 9    # unit range of cis-interaction
    max_span = 32   # 
    n_heads = 8
    in_channels, out_channels = train_dat[0].x.shape[1], 32
    model = GAE(encoder=EdgeConvEncoder(n_heads, in_channels, out_channels, cis_span), decoder=InnerProductDecoder())
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-2)
    criterion = torch.nn.MSELoss()
    # train
    filepath = './trained_model/'
    result = []
    mse = numpy.array([])
    for epoch in range(1,801):
        train(train_loader)
        train_mse, train_cor = test(train_loader) 
        vali_mse, vali_cor = test(vali_loader)
        mse = numpy.append(mse, train_mse.numpy())
        result.append([train_mse, train_cor, vali_mse, vali_cor])
        if (train_cor > 0.6):
            torch.save(model.state_dict(), filepath + '/model_epoch' + str(epoch) + '_Fold' + str(N) + '.pkl')
        #print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}')
        print(f'K-Fold: {N}, Epoch: {epoch:03d}, Train MSE: {train_mse:.4f}, Train cor: {train_cor:.4f}, Test MSE: {vali_mse:.4f}, Test cor: {vali_cor:.4f}')
        pandas.DataFrame([N,epoch,train_mse, train_cor, vali_mse, vali_cor]).T.to_csv(filepath + 'train_log.csv', mode = 'a',index = False, header = False)
        if (train_cor > 0.7) and (len(mse) > 30) and (train_mse.numpy() >= mse[-30:].mean()):
            break
