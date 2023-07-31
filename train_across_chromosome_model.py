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


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')
#print(device)

# In[3]:

for i in range(1,23):
    if os.path.exists('data/chr' + str(i) + '_epi_traindata.pt'):
        train_dataset = torch.load('data/chr' + str(i) + '_epi_traindata.pt')
        test_dataset = torch.load('data/chr' + str(i) + '_epi_testdata.pt')
    else:
        train_dataset = HiCDataset(root='./', name='data', n_seg=128, dataset = 'train', testchr= i, N_chr = 50)
        test_dataset = HiCDataset(root='./', name='data', n_seg=128, dataset = 'test', testchr = i, N_chr = 200)
        torch.save(train_dataset,'data/chr' + str(i) + '_epi_traindata.pt')
        torch.save(test_dataset,'data/chr' + str(i) + '_epi_testdata.pt')
        test_pos = numpy.load("data/processed/test_pos_list.npy")
        train_pos = numpy.load("data/processed/train_pos_list.npy")
        numpy.save('data/chr' + str(i) + '_pos_list.npy',[test_pos, train_pos])
        shutil.rmtree('data/processed/')
    # 随机交换顺序
    n_train = random.sample(range(0,len(train_dataset)), len(train_dataset))
    n_test = random.sample(range(0,len(test_dataset)), len(test_dataset))
    train_dataset = train_dataset[n_train]
    test_dataset = test_dataset[n_test]
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    # model
    cis_span = 9    # unit range of cis-interaction
    max_span = 32   # 
    n_heads = 8
    in_channels, out_channels = train_dataset.num_features, 32
    model = GAE(encoder=EdgeConvEncoder(n_heads, in_channels, out_channels, cis_span), decoder=InnerProductDecoder())
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-2)
    criterion = torch.nn.MSELoss()
    # train
    filepath = './trained_model/'
    result = []
    for epoch in range(1,401):
        train(train_loader)
        train_mse, train_cor = test(train_loader) 
        test_mse, test_cor = test(test_loader)
        result.append([train_mse, train_cor, test_mse, test_cor])
        if (test_cor > 0.6) & (train_cor > 0.6):
            torch.save(model.state_dict(), filepath + '/model_epoch' + str(epoch) + '.chrom' + str(i) + '.pkl')
        #print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}')
        print(f'chrom: {i}，Epoch: {epoch:03d}, Train MSE: {train_mse:.4f}, Train cor: {train_cor:.4f}, Test MSE: {test_mse:.4f}, Test cor: {test_cor:.4f}')
        #pandas.DataFrame([i, epoch,train_mse, train_cor, test_mse, test_cor]).T.to_csv(filepath + 'train_log.csv', mode = 'a',index = False, header = False)
